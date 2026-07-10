## Pipeline to derive burrow masks per video

Steps:

1. Run the bash script at `bash_scripts/compute_prompt_data.sh` to compute the SAM3 prompt data.

    This script will run two scripts:
    a. `scripts/burrows/compute_burrow_prompt_coords.py`, which computes candidate bounding box prompts around trajectory hotspots
    b. `scripts/burrows/compute_burrow_prompt_frames.py`, which computes the frame indices with the lowest detection counts per video (5% of the total frames)

2. Run the bash script at `bash_scripts/compute_min_max_mean_image.sh` to compute the mean frame of the subset of frame indices computes in step 1b.

3. Use `scripts/burrows/annotate_burrow_prompts_manual.py` to generate a set of coordinate prompts from the candidates computed in step 1a.

    This script writes a timestamped `manual_prompt_points_<YYYYMMDD_HHMMSS>.csv` (columns `group_id`, `prompt_point_x`, `prompt_point_y`) into its `--output-dir`; this CSV is the `manual_prompts_csv` input to step 4.


4. Use the script at `scripts/burrows/segment_burrows_sam3.py` to run SAM3 as tiled inference on the mean frames per video and derive the burrow masks.

    Each mean frame is split into overlapping tiles and SAM3 runs per tile using only the prompt points that fall inside that tile (and optionally with a text prompt, by default set to `"hole"`). The per-tile masks are then:
    (1) split into connected regions; (2) filtered by min area; (3) merged across tiles into instances by overlap; and finally (4) filtered by max area and solidity.

    Inputs (positional):
    - `images_dir`: directory of PNG mean frames to segment (from step 2). Each filename's video is the part before the first `_` (and before `-Loop`).
    - `manual_prompts_csv`: the `manual_prompt_points_<...>.csv` from step 3, with columns `group_id` (`<video>_mean_n<frame>.png`), `prompt_point_x`, `prompt_point_y`.
    - `output_dir`: directory the output store is written into.

    Key options: `--text-prompt` (default `"hole"`, pass `""` to disable), `--conf-threshold` (0.35), `--min-mask-area-pixels` (200), `--max-mask-area-pixels` (2500), `--min-solidity` (0.95), `--tile-size` / `--tile-overlap` (default to a third of the image height and half of that), `--overlap-threshold` (0.5), `--max-regions-per-image` (500).

    Optional HTML plots: pass `--save-html-plots` to also write one self-contained interactive Plotly HTML plot per frame into `output_dir` (a timestamped `plots_<masks_zarr_stem>_<...>/` subdirectory), overlaying the colored mask instances and manual point prompts on the frame with the mask ID and SAM3 score on hover and toggleable layers. Add `--trajectories-zarr <CrabTracks.zarr>` (optionally with `--min-hits-per-burrow-frac`, default `0.10`) to also overlay the datashader crab-trajectory raster and red contours around high-activity burrows.

    Output: a single timestamped zarr store `output_dir/masks_<YYYYMMDD_HHMMSS>.zarr` (timestamped so runs don't collide), laid out with one group per video (each mean frame maps to one video):
    ```
    masks_<YYYYMMDD_HHMMSS>.zarr
    └── <video>
        ├── masks   (H, W) int16               — ID-encoded mask (background = 0, instances = 1, 2, ...)
        └── scores  (max_regions + 1,) float32 — SAM3 score per instance ID
    ```
    - `<video>/masks` (H, W) int16 — the ID-encoded mask for that video's mean frame.
    - `<video>/scores` (max_regions + 1,) float32 — SAM3 score indexed by instance ID (`scores[i]` is the score of mask instance `i`; index 0 corresponds to the background, which is unused and set to NaN).
    - root attributes record some of the run parameters (model, source paths, tile geometry, postprocessing thresholds, encoding) and per-run metrics (`frames_with_masks`, `n_masks_per_frame`).
