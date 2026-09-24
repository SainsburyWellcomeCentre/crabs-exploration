# Tracker

We currently use [SORT](https://github.com/abewley/sort), an algorithm based on Kalman filtering, to track the detected crabs across frames. As stated in the SORT repository, this tracker doesn't handle occlusions or re-entering of objects, and it was developed mainly as a baseline and testbed for the development of future trackers.

The configurable parameters of the tracker are defined in `crabs-exploration/crabs/tracker/config/tracking_config.yaml`:

- `iou_threshold`: defines the minimum IOU value between a ground truth box and a detection, to consider a detection a true positive. By default, 0.1.
- `score_threshold`: defines the minimum confidence score for a detection to be considered for tracking. By default, 0.1.
- `max_age`: maximum number of frames to keep a track "alive" without associated detections. By default, 10.
- `min_hits`: minimum number of detections required to initialise a track. By default, 1.

## Evaluation

We evaluate the performance of the tracker against manually labelled ground-truth. This ground-truth consists of manually annotated bounding boxes and IDs. We use MOTA (Multiple Object Tracking Accuracy) as a metric to evaluate performance. For each frame in the manually labelled clip, we can compute MOTA as:

```
MOTA = 1 - ((FN + FP + IDS) / GT)
```

where `FN` is the number of false negatives (missed detections), `FP` is the number of false positives, `IDS` is the number of identity switches, and `GT` is the total number of ground-truth objects. The higher the MOTA value, the better the tracking performance. Note that the MOTA metric is upper-bounded by 1, and lower-bounded by -Inf. For a full video clip, we compute the MOTA per frame, and report the average MOTA across all frames.

To compute the total number of false negatives (or missed detections, `FN`) at a given frame `f`, we count the number of ground-truth objects that do not match with any detection at frame `f`. A ground-truth object and a detection are considered to match if their associated boxes sufficiently overlap, that is, if their intersection-over-union (IOU) is greater than a given threshold.

To compute the total number of false positives (`FP`) at a given frame `f`, we count the number of detections that do not match with any of the ground-truth object defined at frame `f`.

A true positive (`TP`) is defined as a detection that sufficiently overlaps with a ground-truth box (with overlap measured with the `IOU` metric).

To compute the number of identity switches (`IDS`) we inspect the set of true positives. Given two mappings from ground-truth IDs to predicted IDs, for the previous frame `f-1` and for the current frame `f`, we compute the total number of identity switches (`IDS`) as the sum of:
- the number of **re-identifications**, that is, the number of times the same ground-truth ID maps to two different predicted IDs in the current and the previous frame. If the predicted ID in the previous frame is not defined (because it was a missed detection or because there was no ground-truth defined for it), we use the last predicted ID associated to that ground-truth ID if available.
- the number of **identity swaps**, that is, the number of times the same predicted ID maps to two different ground-truth IDs in the current and previous frame.

Note that this definition of identity switches is slightly different to some other MOTA definitions, which only account for identity switches between consecutive frames. It is also different from other implementations, which define an "expected" predicted ID for each ground-truth ID. This "expected" predicted ID is the predicted ID that is most often (in terms of number of frames) associated to a ground-truth ID.

## Masking tracked crabs

`mask-tracked-crabs` prompts [SAM2](https://github.com/facebookresearch/sam2) with boxes someone already computed, and writes a zarr store holding **one label image per frame**: a single `uint16` array in which each crab's pixels carry that crab's own value, and `0` is background. It needs no trained detector and no detection pass.

SAM2 is an opt-in dependency, since it is not on PyPI:

```bash
uv sync --group masks
# or, in a conda environment with torch already installed:
pip install --no-build-isolation "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
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

The knobs are in `crabs/tracker/config/mask_config.yaml`, and a config of your own need only name the keys it changes:

- `sam2_model_id`: the Hugging Face model. By default, `facebook/sam2.1-hiera-base-plus`.
- `max_prompts_per_batch`: how many box prompts go to SAM2 at once. It bounds peak memory, because every prompt gets a full-frame float32 mask back. By default, 32.
- `shard_n_frames`: frames per shard file. It sets both the write buffer (`shard_n_frames × H × W × 2` bytes) and the file count, and is the one to tune per filesystem. By default, 32.
- `occlusion_policy`: which crab keeps a pixel where two masks overlap. By default, `smallest_wins`, which is the only policy implemented.

### The store, and how to read it

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

## References and useful resources

- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP) (pp. 3464-3468). IEEE. [link](https://arxiv.org/abs/1602.00763)
- Bernardin, K., & Stiefelhagen, R. (2008). Evaluating multiple object tracking performance: the clear mot metrics. EURASIP Journal on Image and Video Processing, 2008, 1-10. [link](https://link.springer.com/article/10.1155/2008/246309)
- Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129, 548-578. [link](https://link.springer.com/article/10.1007/s11263-020-01375-2)
- [TrackEval library](https://github.com/JonathonLuiten/TrackEval)
- py-motmetrics library
- [MOTChallenge Evaluation Kit](https://github.com/dendorferpatrick/MOTChallengeEvalKit)
- Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016, October). Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17-35). Cham: Springer International Publishing. [link](https://arxiv.org/abs/1609.01775) - might be useful for multi-camera tracking evaluation.
