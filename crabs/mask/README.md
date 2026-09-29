# Masking tracked crabs

`mask-tracked-crabs` prompts [SAM2](https://github.com/facebookresearch/sam2) with boxes someone already computed, and writes a zarr store holding **one label image per frame**: a single `uint16` array in which each crab's pixels carry that crab's own value, and `0` is background. It needs no trained detector and no detection pass.

SAM2 and its dependencies are in the opt-in `masks` extra. SAM2's build imports torch, so install the package first and then the extra:

```bash
uv sync
uv sync --extra masks
# or, with pip:
pip install .
pip install setuptools setuptools_scm  # needed below, as build isolation is off
pip install --no-build-isolation ".[masks]"
```

```bash
mask-tracked-crabs \
    --boxes CrabTracks-slurm3012633.zarr \
    --videos /path/to/loop-clips/ \
    --match "04.09.2023*" \
    --output_dir mask_output
```

- `--boxes` is a trajectories zarr store written by `create-zarr-dataset`. Its suffix selects how it is read.
- `--videos` is the directory holding the clip videos the boxes refer to, named `<video-id>-<clip-id>.mp4` — the files `extract-loop-clips` writes.
- `--match` is a glob over **video groups**, not clips; every clip of a selected video is masked.
- The store name carries a timestamp, so re-masking with a different SAM2 model does not collide. Readers glob for it rather than naming it.

To mask a whole trajectories store on the cluster, use [`bash_scripts/run_mask_array.sh`](../../bash_scripts/run_mask_array.sh): a SLURM array with one task per video, each appending its own group to one shared store. See [`guides/MaskTrackedCrabsHPC.md`](../../guides/MaskTrackedCrabsHPC.md).

The knobs are in `crabs/mask/config/mask_config.yaml`, and a config of your own need only name the keys it changes:

- `sam2_model_id`: the Hugging Face model. By default, `facebook/sam2.1-hiera-base-plus`.
- `max_prompts_per_batch`: how many box prompts go to SAM2 at once. It is the main knob for peak GPU memory, because SAM2 upsamples every prompt's mask to the full frame as float32 logits and then thresholds it to a boolean mask (`H × W × 5` bytes, ~44 MB at 4096×2160, so ~1.4 GB for 32 prompts). By default, 32.
- `shard_n_frames`: frames per shard file. Finished frames are held in memory until a shard is full, then written as one file, so it sets both the write buffer (`shard_n_frames × H × W × 2` bytes, ~570 MB for 32 frames at 4096×2160) and the file count. Larger values use more memory but produce fewer files; it is the one to tune per filesystem. `null` disables sharding, which is like `1` but with no shard index. By default, 32.
- `occlusion_policy`: which crab keeps a pixel where two masks overlap. By default, `smallest_wins`, which is the only policy implemented.

## The store, and how to read it

```
<output_dir>/<name>_masks_<timestamp>.zarr/
└── <video_id>/                                  # one group per video
    ├── labels    (clip_id, time, img_h, img_w)  uint16
    ├── label_of  (individual,)                  uint16
    ├── clip_id     e.g. ["Loop00", "Loop05"]
    └── individual  e.g. ["id_0000", …]  — copied from the trajectories store
```

The `clip_id`, `time` and `individual` coordinates are the trajectories store's own, so masks and trajectories for the same clips align 1:1 with no reindexing. A label image is the native input of both consumers, so there is no conversion step on read:

```python
import xarray as xr
from skimage.measure import regionprops

ds = xr.open_datatree(store, engine="zarr", chunks={})["<video_id>"]

v = int(ds.label_of.sel(individual="id_0003"))         # this crab's pixel value
mask = ds.labels.sel(clip_id="Loop05") == v            # (time, img_h, img_w) bool
regionprops(ds.labels.sel(clip_id="Loop05").isel(time=t).values)
viewer.add_labels(ds.labels)                           # napari, sliders over clip and time
```

Five things the store cannot say for itself:

1. **⚠️ Occlusion is resolved at write time, and the losses are not recorded.** Where two masks overlapped, the smaller crab kept the contested pixels and the larger one's are *gone from the store*. Measured on the trajectories store: 94% of crabs never overlap and 0.62% of box area is contested, but the tail is heavy — 2.4% of crabs lose more than a quarter of their box.

    Filtering occluded crabs is therefore a read-side job: **by adjacency**, since two labelled regions that touch in `labels` were plausibly overlapping (this over-flags, but needs nothing extra); or **by area**, comparing a crab's `regionprops` area against its tracked box area from the trajectories store, which is already aligned. The orientation work should exclude flagged instances rather than fit ellipses to remnants. Recovering the exact loss would mean re-running SAM2.

2. **`individual` names mean something only within a clip.** `id_0003` in `Loop00` and in `Loop01` are different crabs, because individuals are renumbered from zero per clip. Always select a `clip_id` before an `individual`.

3. **And only within a video.** The names are not comparable across video groups, and their *format* differs too: some videos use `id_000` and others `id_0000`, because the padding is derived from each video's own number of individuals. **Never rebuild a name with a format string** — read `ds.individual`.

4. **`label_of` is the mapping, in both directions.** There is no offset to remember. Going back from one pixel value is a search, `str(ds.individual.values[ds.label_of.values == v][0])`. Decoding a whole `regionprops` output is where an inverse array earns its place, built once and hoisted out of the sweep:

    ```python
    inv = np.empty(int(ds.label_of.max()) + 1, dtype=int)
    inv[ds.label_of.values] = np.arange(ds.sizes["individual"])
    ds.individual.values[inv[[p.label for p in props]]]     # -> ['id_0002', 'id_0000', …]
    ```

5. **The trajectories store is the index for the mask store.** `labels` has no `individual` axis, so fetching one crab is a full-clip scan unless you narrow it first. `position` is NaN where a crab is absent, so `isel(time=frames_present)` cuts a real example from 27,054 frames to 1,158.
