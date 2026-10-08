# Masking tracked crabs

`mask-tracked-crabs` runs [SAM2](https://github.com/facebookresearch/sam2) to predict masks from input bounding boxes, and writes a zarr store holding **one label image per frame**. A label image is a single `uint16` array in which each crab's pixels carry that crab's own value, and `0` is background.

## Installation
SAM2 and its dependencies are in the opt-in `masks` extra. SAM2's build imports torch, so to use the torch version specified by `crabs` we install the package first and then the extra:

```bash
uv sync
uv sync --extra masks
# or, with pip:
pip install .
pip install setuptools setuptools_scm  # needed below, as build isolation is off
pip install --no-build-isolation ".[masks]"
```

## Segmentation
Once installed, we can run SAM2 prediction as follows:
```bash
mask-tracked-crabs \
    --boxes CrabTracks-slurm3012633.zarr \
    --videos /path/to/loop-clips/ \
    --match "04.09.2023*" \
    --output_dir mask_output \
    --mask_config_file my_mask_config.yaml  # optional
```

- `--boxes` is a trajectories zarr store written by `create-zarr-dataset`.
- `--videos` is the directory holding the clip videos the boxes refer to. These are named `<video-id>-<clip-id>.mp4`, and are the files `extract-loop-clips` writes.
- `--match` is a glob over **video groups**, not clips. It allows us to select a subset of videos, but every clip of a selected video is masked.
- `--mask_config_file` is optional: a YAML file with masking parameters that differ from the defaults (see [Parameters](#parameters)).
- The output store name carries a timestamp, so re-masking does not overwrite the store.

To run the masking pipeline over a full trajectories store on the cluster, use the script provided at [`bash_scripts/run_mask_array.sh`](../../bash_scripts/run_mask_array.sh). It will launch a SLURM array job, with each job processing one video and appending its result to the shared store. See [`guides/MaskTrackedCrabsHPC.md`](../../guides/MaskTrackedCrabsHPC.md) for further details.

## Parameters
The default masking parameters live in [`crabs/mask/config/mask_config.yaml`](config/mask_config.yaml). To set non-default values, copy that file, edit it, and pass it with `--mask_config_file`. Your file only needs the keys you want to change; the rest take the defaults listed below.

- `sam2_model_id`: the SAM2 model to use, as named by Hugging Face. By default, `facebook/sam2.1-hiera-base-plus`. The options are, from fastest to most accurate:
    - `facebook/sam2.1-hiera-tiny`
    - `facebook/sam2.1-hiera-small`
    - `facebook/sam2.1-hiera-base-plus`
    - `facebook/sam2.1-hiera-large`

    The older `facebook/sam2-hiera-*` checkpoints also work.
- `max_prompts_per_batch`: how many box prompts are passed to SAM2 in one prediction step (by default 32). It is the main knob for controlling peak GPU memory. A prediction step covers one frame at most. SAM2 upsamples every prompt's mask to the full frame as float32 logits and then thresholds it to a boolean mask, which costs `H × W × 5` bytes per prompt (~44 MB at 4096×2160, so ~1.4 GB for 32 prompts).
- `n_frames_per_shard`: frames per shard file (by default 32). Finished frames are held in memory until a shard is full, then they are written as one file. Larger values use more memory but produce fewer files. A `null` value disables sharding, which is like `1` but with no shard index. For 32 frames per shard at 4096×2160, the in-memory buffer will be `n_frames_per_shard × H × W × 2` bytes, ~570 MB.
- `occlusion_policy`: which crab keeps a pixel where two masks overlap. By default, `smallest_wins`, which is the only policy implemented.


> [!NOTE]
> **Chunks and shards.** A **chunk** is the unit zarr compresses and reads. To read any pixel, zarr has to decompress the whole chunk that contains it, so the chunk size is chosen to match the most common read pattern (here, by frame, so each `labels` chunk is `(1, 1, img_h, img_w)`). A **shard** is a file on disk that holds many chunks, plus an index saying where each chunk sits in the file. The index allows one chunk to still be read on its own. Sharding is useful to keep the number of files manageable on a filesystem.

## Output store format

```
<output_dir>/<name>_masks_<timestamp>.zarr/
└── <video_id>/                                  # one group per video
    ├── labels    (clip_id, time, img_h, img_w)  uint16
    ├── label_of  (individual,)                  uint16
    ├── clip_id     e.g. ["Loop00", "Loop05"]
    └── individual  e.g. ["id_0000", …]  — copied from the trajectories store
```

The `clip_id`, `time` and `individual` coordinates are the trajectories store's own, so masks and trajectories for the same clips align 1:1 with no reindexing. `label_of` maps each `individual` to its pixel value in the label image, and the reverse lookup for a label `lbl` is `ds.individual.values[ds.label_of.values == lbl]`.

> [!TIP]
> `labels` has no `individual` axis, so getting one crab's mask means reading every frame of the clip. To speed this up, first use the trajectories store to find the frames where that crab's `position` is not NaN, and read only those (in one example, 1,158 of 27,054 frames).

A reminder that in both the masks and the trajectories store, `individual` names are not consistent across videos and refer only to a clip. Their padding may also differ between videos (`id_000` vs `id_0000`), so read them from `ds.individual` rather than building them with a format string.

A label image is easily consumed by `scikit-image`'s `regionprops`:

```python
import xarray as xr
from skimage.measure import regionprops

ds = xr.open_datatree(store, engine="zarr", chunks={})["<video_id>"]

# Compute region properties for crab with individual="id_0003" in Loop05
mask_label = int(ds.label_of.sel(individual="id_0003"))
mask = ds.labels.sel(clip_id="Loop05") == mask_label     # (time, img_h, img_w) bool
regionprops(ds.labels.sel(clip_id="Loop05").isel(time=t).values)
```

To display the labels of all clips in one video in `napari`:
```python
import napari

viewer = napari.Viewer()
viewer.add_labels(ds.labels, name="crab masks")  # adds sliders over clip_id and time
napari.run()  # needed when running from a script; not in a notebook or IPython
```
