"""Segment burrows with SAM3 from per-video point prompts (tiled inference).

For every PNG frame in the input directory, we look up the manual point prompts
for its video and run SAM3 image inference **tile by tile**: the frame is split
into overlapping tiles and SAM3 (optionally with a text prompt) runs on each
tile with only the prompt points that fall inside it, re-normalised to the
tile's coordinates. Tiles with no prompt point are skipped. Running on smaller
tiles helps SAM3 resolve small burrows that full-image inference misses.

Each tile's predicted masks are split into connected regions and a cheap
minimum-area filter drops speckle; the surviving regions are mapped back onto
the full-image canvas.

The regions (pooled across all tiles of a frame) are merged into instances by
overlap: two regions are fused when their intersection covers at least a
threshold fraction of the smaller region's area (merging is transitive). Each
instance inherits the max score among its contributors. The assembled instances
are then filtered by area (upper bound) and solidity -- applied here, after the
merge, so a burrow clipped at a tile seam is judged as one whole blob rather
than as its more-convex fragments (which a per-tile filter would preferentially
keep, leaving the whole burrow split across instances). Surviving instances are
written into a per-video group of a timestamped zarr store as an ID-encoded
mask (background = 0, regions = 1, 2, ...), alongside a sibling array with the
SAM3 score per region.

The store is timestamped and laid out with one group per video:
    masks_<YYYYMMDD_HHMMSS>.zarr
    └── <video>
        ├── masks   (H, W) int16              # ID-encoded mask
        └── scores  (max_regions + 1,) float32  # score per region ID
                                                # (index 0 = background, NaN)

The input prompt CSV is the one produced by annotate_burrow_prompts_manual.py,
with columns:
    group_id,        # "<video>_mean_n<frame>.png"; video = part before "_mean"
    prompt_point_x,
    prompt_point_y

Usage (dependencies are auto-installed via uv):
* Default (text prompt "hole")
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir
* Disable the text prompt (rely on point prompts only)
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir --text-prompt ""
* Custom postprocessing thresholds
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir \
        --min-mask-area-pixels 100 --max-mask-area-pixels 3000
* Custom tile geometry (defaults derive from image height)
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir \
        --tile-size 1024 --tile-overlap 512
* Also export an interactive HTML plot per frame (masks + prompts + scores)
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir --save-html-plots
* HTML plots with the crab-trajectory raster and activity contours overlaid
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir --save-html-plots \
        --trajectories-zarr /path/to/CrabTracks.zarr

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "pycocotools",
#   "sam3 @ git+https://github.com/facebookresearch/sam3.git",
#   "einops",
#   "huggingface_hub",
#   "scikit-image",
#   "numpy",
#   "pandas",
#   "psutil",  # undeclared transitive dep of sam3's video predictor
#   "plotly",  # optional HTML plot export (--save-html-plots)
#   "datashader",  # optional trajectory raster layer (--trajectories-zarr)
#   "xarray",  # optional trajectory datatree (--trajectories-zarr)
#   "dask",  # required by xarray to open the trajectory datatree lazily
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

import argparse
import contextlib
import os

# Reduce CUDA allocator fragmentation. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.feature import peak_local_max
from skimage.measure import find_contours, regionprops
from skimage.measure import label as sk_label
from skimage.segmentation import watershed

# matplotlib's tab10 palette as (10, 3) uint8 RGB, cycled over mask IDs when
# rendering the HTML overlays (hardcoded to avoid a matplotlib dependency).
TAB10_RGB = np.array(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ],
    dtype=np.uint8,
)


class ImageArrayLazy:
    """A lazy array for images passed as a list."""

    def __init__(self, img_paths: list[Path]):
        """Store sorted image paths and cache the shared image shape."""
        # add sorted list of paths
        self.img_paths = sorted(img_paths)

        # add image shape, assuming all have same as first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """Load and return the image at ``idx`` as an array."""
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        """Return the stack shape (B, H, W, C)."""
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)


def _initialise_mask_zarr(
    output_dir: Path,
    metadata_dict: dict,
) -> tuple[zarr.Group, Path]:
    """Create an empty timestamped zarr store with metadata.

    The per-video ``masks`` and ``scores`` arrays are created lazily inside the
    inference loop (one group per video). Mask instance IDs start at 1, since 0
    is reserved for the background label, so each video's ``scores`` array has
    ``max_regions_per_image + 1`` entries (index 0, the background, is unused
    and left as NaN).
    """
    # Create a timestamped masks zarr store in the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_masks_zarr = output_dir / f"masks_{timestamp}.zarr"

    # Initialise root and attach metadata; per-video arrays added downstream.
    root = zarr.open_group(output_masks_zarr, mode="w")
    root.attrs.update(metadata_dict)

    return root, output_masks_zarr


def _extract_point_prompts_per_video(
    manual_prompts_csv: Path | str,
) -> dict:
    """Compute dict mapping video to point prompt pixel coordinates.

    The CSV ``group_id`` column has format ``<video>_mean_n<frame>.png``; the
    video string is everything before the first underscore. Coordinates are the
    pixel (x, y) prompt locations as stored in the CSV. Tiled inference
    re-normalises these to each crop's size downstream, so no full-image
    normalisation is applied here.
    """
    # Read from csv
    df_prompts = pd.read_csv(manual_prompts_csv)

    # point prompts in pixel xy, keyed by video string
    points_xy_px_per_video = {
        str(video).split("_")[0]: group[["prompt_point_x", "prompt_point_y"]]
        .to_numpy()
        .astype(np.float32)
        for video, group in df_prompts.groupby("group_id")
    }

    return points_xy_px_per_video


def _add_prompts_to_inference_state(
    processor, image, prompts_xy_norm, text_features=None
):
    """Add image, normalised points and optional text features to the state.

    ``text_features`` is the precomputed output of
    ``model.backbone.forward_text([prompt])`` (image-independent, so encoded
    once and reused across tiles/frames). When provided, it is injected into
    the state directly instead of calling ``processor.set_text_prompt``, which
    would re-encode the text and run an extra grounding pass with only the
    dummy geometric prompt (immediately discarded once the points are added).
    Pass ``None`` to rely on the point prompts only.
    """
    # Pass image to processor and reset inference state
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(
        inference_state
    )  # mutates the state dict in place

    # Inject the precomputed text features (mirrors set_text_prompt minus the
    # text encoder pass and the premature grounding pass)
    if text_features is not None:
        inference_state["backbone_out"].update(text_features)

    # Add normalised point prompts (all at once; grounds a single time)
    inference_state = _add_point_prompts(
        processor, inference_state, prompts_xy_norm, labels=True
    )

    return inference_state


def _add_point_prompts(processor, inference_state, points_xy, labels=True):
    """Add point prompts to the inference state.

    This function mirrors ``Sam3Processor.add_geometric_prompt`` but appends
    points as geometric prompts instead of a bounding box. It relies on SAM3
    internals (``_get_dummy_prompt`` and ``_forward_grounding``) as the
    processor does not expose a public point method.

    All points are appended and the grounding forward pass runs **once** at
    the end (rather than once per point). Re-running ``_forward_grounding``
    after every point recomputes full-resolution masks for all accumulated
    objects each time, which is wasteful and blows up GPU memory.

    Note: ``points_xy`` is an ``(N, 2)`` array of ``(x, y)`` pairs normalized
    to ``[0, 1]`` (a single ``(x, y)`` pair is also accepted). ``labels``
    sets each input point as positive (``True``, default) or negative
    (``False``); pass a scalar to apply the same label to all points, or an
    ``(N,)`` sequence for per-point labels.
    """
    # Throw error if image not passed thru backbone yet
    # (if passed, SAM3 stashes the backbone resulting features under the key
    # "backbone_out" in the state dict; we need an image to ground against).
    if "backbone_out" not in inference_state:
        raise ValueError("call processor.set_image before adding a prompt")

    # Check if a text prompt was passed
    if "language_features" not in inference_state["backbone_out"]:
        # if not: set text prompt to a dummy "visual" text so the
        # model relies only on the geometric prompt (this is the standard
        # way SAM3 signals this, see `add_geometric_prompt`)
        dummy_text_outputs = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        inference_state["backbone_out"].update(dummy_text_outputs)

    # Ensure a geometric_prompt container exists to append the point into.
    # (we need to initialise it here because we append to it later)
    if "geometric_prompt" not in inference_state:
        inference_state["geometric_prompt"] = (
            processor.model._get_dummy_prompt()
            # returns an empty "Prompt" container with number of boxes = 0;
            # an empty Prompt container (zero boxes, batch=1) that subsequent
            # append_boxes/append_points calls grow into.
            #
            # Note: the SAM3 image processor also supports mask prompts
            # internally but this is not exposed in the public API. Masks as
            # prompts are used internally by its tracker/video stack, when the
            # previous frame mask is passed as prompt to the next frame.
        )

    # Add input point prompts with their labels to the inference state
    # NOTE: here batch=1, n_points = N (the number of points passed in)
    points_xy = np.atleast_2d(
        np.asarray(points_xy, dtype=np.float32)
    )  # (N, 2)
    n_points = points_xy.shape[0]

    point_coords = torch.tensor(
        points_xy,
        device=processor.device,
        dtype=torch.float32,
    ).view(n_points, 1, 2)  # (n_points, batch, 2)

    point_labels = torch.as_tensor(
        np.broadcast_to(labels, (n_points,)).copy(),
        device=processor.device,
        dtype=torch.bool,
    ).view(n_points, 1)  # (n_points, batch)

    inference_state["geometric_prompt"].append_points(
        point_coords,
        point_labels,
    )

    # Ground once over all accumulated prompts.
    return processor._forward_grounding(inference_state)


def _extract_sam3_results_one_img(inference_state):
    """Extract SAM3 boolean masks and scores for a single image.

    Masks are moved to CPU so this frame's GPU state can be released before
    the next frame: otherwise the previous state (backbone features +
    full-res masks_logits) stays alive during the next forward pass.
    """
    masks = inference_state["masks"].cpu().numpy()
    scores = inference_state["scores"].float().cpu().numpy()  # (N,)

    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    n_objects = masks.shape[0]

    return masks, scores, n_objects


def _make_tiles(
    img_h: int, img_w: int, tile_side: int, tile_overlap: int
) -> list[tuple[int, int, int, int]]:
    """Overlapping (x0, y0, x1, y1) tiles covering the image.

    Standard image coordinates (origin top-left, y increasing downward). Tiles
    are clamped to the image edges and derived from their bottom-right corner
    so that every tile stays full-sized (edge tiles shift inward rather than
    shrink). ``tile_overlap`` is the overlap in pixels between neighbours; aim
    for it to exceed a burrow's diameter so each burrow is fully contained in
    at least one tile.
    """
    step = tile_side - tile_overlap

    # we use a set to easily remove tuple duplicates (near the edge, the
    # clamping behaviour means many tiles can be "rounded" to the same tile)
    tiles = set()

    # loop thru candidate top-left corner of tile
    for y0 in range(0, max(1, img_h - tile_overlap), step):
        for x0 in range(0, max(1, img_w - tile_overlap), step):
            # compute bottom-right corner of tile, clamped to image edge
            x1 = min(x0 + tile_side, img_w)
            y1 = min(y0 + tile_side, img_h)

            tiles.add(
                (
                    # top-left corner derived from the bottom-right corner so
                    # the tile stays full-sized (matches x0,y0 in the interior)
                    max(0, x1 - tile_side),
                    max(0, y1 - tile_side),
                    x1,
                    y1,
                )
            )
    return sorted(tiles)


def _filter_masks_into_regions(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int,
):
    """Split masks into connected regions and drop those below ``min_area``.

    ``masks`` is (N, H, W) boolean and ``scores`` is (N,) aligned with it. Each
    mask is split into connected regions (a single SAM3 mask isn't guaranteed
    to be one clean blob); a region is kept only if its area is at least
    ``min_area``.

    This is a cheap per-tile speckle filter run *before* the overlap merge. The
    upper-area and solidity filters are deliberately applied afterwards, on the
    assembled instances (see ``_filter_instances_by_area_solidity``): a burrow
    clipped at a tile seam splits into smaller, more-convex fragments, so a
    per-tile ``max_area``/solidity filter would preferentially keep those
    fragments and drop the whole burrow, leaving it split across instances.
    ``min_area`` is safe here because a fragment is never larger than the whole
    burrow it came from.

    Returns ``(kept_regions_bool_masks, kept_regions_scores, drop_counts)``
    where ``kept_regions_bool_masks`` is a list of (H, W) boolean masks (in the
    coordinate frame of the input ``masks``), ``kept_regions_scores`` carries
    the source mask's score onto each kept region, and ``drop_counts`` counts
    how many connected regions were dropped per reason.
    """
    kept_regions_bool_masks = []
    list_mask_idcs = []
    drop_counts = {"area_low": 0}

    # Loop thru masks
    for mask_idx, mask in enumerate(masks.astype(bool)):
        # Label connected regions in mask
        # (connected regions are assigned the same int)
        # SAM3 returns a boolean mask, but a single mask isn't
        # guaranteed to be one clean blob
        label_mask = sk_label(mask)

        # Loop thru regions
        for region in regionprops(label_mask):
            # Filter by min area (cheap speckle removal before the merge)
            if region.area < min_area:
                drop_counts["area_low"] += 1
                continue

            # If it passes: retain that region within the mask
            kept_regions_bool_masks.append(label_mask == region.label)
            # Keep track of the mask ID associated to this region too
            list_mask_idcs.append(mask_idx)

    # Get list of scores for kept regions
    kept_regions_scores = scores[list_mask_idcs]

    return kept_regions_bool_masks, kept_regions_scores, drop_counts


def _merge_bool_masks_by_connectivity(
    list_kept_region_masks, list_scores, img_h_w, connectivity
):
    """Merge boolean region masks into one ID-encoded mask by connectivity.

    All region masks are OR-ed into a single binary canvas, then connected
    components are labelled (``skimage.measure.label``) so that overlapping,
    nested or touching regions collapse into a single instance. The resulting
    labels are dense (``1..M``, with 0 = background).

    Each merged instance inherits the **maximum** score among the regions that
    contributed to it. Returns ``(id_encoded_mask, surviving_ids,
    surviving_scores)`` with ``surviving_ids`` ascending (``1..M``) and
    ``surviving_scores`` index-aligned to them.

    ``connectivity`` is passed to ``skimage.measure.label``: 1 = orthogonal
    neighbours only (4-connectivity), 2 = include diagonals (8-connectivity).
    """
    # Paint all regions into a single binary canvas (their union)
    canvas = np.zeros(img_h_w, dtype=bool)
    for bool_mask in list_kept_region_masks:
        canvas |= bool_mask

    # Label connected components; sk_label yields dense labels 1..M
    id_encoded_mask = sk_label(canvas, connectivity=connectivity).astype(
        np.int16
    )
    surviving_ids = np.unique(id_encoded_mask)
    surviving_ids = surviving_ids[surviving_ids != 0]

    # Compute merged scores as the max of contributing regions.

    # one score per label ID, excluding 0
    # (every label has at least 1 contributing region, so np.inf should not
    # leak into the zarr)
    surviving_scores = np.full(len(surviving_ids), -np.inf, dtype=np.float32)
    for bool_mask, score in zip(
        list_kept_region_masks, list_scores, strict=True
    ):
        # get *merged* label ID for this *unmerged* mask
        # (i.e. get the final ID for this region)
        label = id_encoded_mask[bool_mask][0]
        # get the max between this unmerged mask's score and the score of the
        # final mask it contributes to
        surviving_scores[label - 1] = max(surviving_scores[label - 1], score)

    return id_encoded_mask, surviving_ids, surviving_scores


def _merge_bool_masks_by_overlap(
    list_kept_region_masks, list_scores, img_h_w, overlap_threshold
):
    """Merge boolean region masks that sufficiently overlap into instances.

    Two regions are merged when their intersection covers at least
    ``overlap_threshold`` of the *smaller* region's area, i.e. when
    ``intersection / min(area_i, area_j) >= overlap_threshold``. Merging is
    transitive: if A overlaps B and B overlaps C, all three collapse into a
    single instance even if A and C do not directly overlap (connected
    components of the overlap graph).

    Each merged instance is the union of its member regions and inherits the
    **maximum** score among them. Pixels shared by regions that were *not*
    merged (overlap below the threshold) are assigned to the higher-scoring
    instance. Returns ``(id_encoded_mask, surviving_ids, surviving_scores)``
    with ``surviving_ids`` ascending and ``surviving_scores`` index-aligned.
    """
    n_regions = len(list_kept_region_masks)

    # Handle the empty case
    if n_regions == 0:
        return (
            np.zeros(img_h_w, dtype=np.int16),
            np.array([], dtype=np.int16),
            np.array([], dtype=np.float32),
        )

    scores = np.asarray(list_scores, dtype=np.float32)

    # Flatten each region to its true-pixel indices and assemble a sparse
    # (n_regions, n_pixels) matrix. Regions are small relative to the frame,
    # so this is far cheaper in memory than a dense equivalent.
    n_pixels = int(np.prod(img_h_w))
    flat_indices = [np.flatnonzero(m.ravel()) for m in list_kept_region_masks]
    indptr = np.concatenate(
        [[0], np.cumsum([len(idx) for idx in flat_indices])]
    )
    indices = np.concatenate(flat_indices)
    flat = csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(n_regions, n_pixels),
    )

    # Pairwise intersection counts (n_regions, n_regions); the diagonal holds
    # each region's own area (sum of 1s).
    intersection = (flat @ flat.T).toarray()
    areas = intersection.diagonal()

    # Overlap = intersection / min(area_i, area_j). Two regions are adjacent
    # when this reaches the threshold (self-pairs on the diagonal excluded).
    # Areas are >= min_area (filtered upstream), so min_areas is never zero.
    min_areas = np.minimum.outer(areas, areas)
    adjacency = (intersection / min_areas) >= overlap_threshold
    np.fill_diagonal(adjacency, False)

    # Merge transitively: connected components of the overlap graph.
    n_components, comp_labels = connected_components(
        csr_matrix(adjacency), directed=False
    )

    # Score per component = max score among its member regions.
    comp_scores = np.full(n_components, -np.inf, dtype=np.float32)
    np.maximum.at(comp_scores, comp_labels, scores)

    # Paint regions into the ID-encoded mask (component label + 1, since 0 is
    # the background). Paint in ascending component-score order so pixels
    # contested by unmerged regions go to the higher-scoring instance.
    id_encoded_mask = np.zeros(img_h_w, dtype=np.int16)
    for region_idx in np.argsort(comp_scores[comp_labels]):
        id_encoded_mask[list_kept_region_masks[region_idx]] = (
            comp_labels[region_idx] + 1
        )

    # Derive surviving IDs from the painted mask (a component fully hidden by
    # higher-scoring overlaps would not appear) and align their scores.
    surviving_ids = np.unique(id_encoded_mask)
    surviving_ids = surviving_ids[surviving_ids != 0]
    surviving_scores = comp_scores[surviving_ids - 1].astype(np.float32)

    return id_encoded_mask, surviving_ids, surviving_scores


def _split_bool_masks_by_watershed(
    list_kept_region_masks, list_scores, img_h_w, min_peak_distance
):
    """Project masks and then split using watershed."""
    # Compute score_canvas: score per pixel of unmerged regions
    score_canvas = np.full(img_h_w, -np.inf, dtype=np.float32)
    for bool_mask, score in zip(
        list_kept_region_masks, list_scores, strict=True
    ):
        np.maximum(score_canvas, score, where=bool_mask, out=score_canvas)

    # Collapse all boolean masks into one canvas
    canvas = np.zeros(img_h_w, dtype=bool)
    for bool_mask in list_kept_region_masks:
        canvas |= bool_mask

    # Split masks using watershed (returns an ID-encoded mask)
    # compute peaks of distance to background (0-value pixels)
    distance = ndi.distance_transform_edt(canvas)
    coords = peak_local_max(
        distance, labels=canvas, min_distance=min_peak_distance
    )
    # express peaks as markers in a zero array, each with an
    # ID assigned
    markers = np.zeros(img_h_w, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    id_encoded_mask = watershed(-distance, markers, mask=canvas).astype(
        np.int16
    )

    # Compute surviving ids
    surviving_ids = np.unique(id_encoded_mask)
    surviving_ids = surviving_ids[surviving_ids != 0]

    # Compute score per final mask as the max score of original masks
    # that make it.
    # Every basin pixel is inside the union, so score_canvas is finite
    # -> no -inf leaks.
    surviving_scores = ndi.maximum(
        score_canvas,  # scores of unmerged masks
        labels=id_encoded_mask,  # labels of final masks
        index=surviving_ids,  # get one score per final mask
    ).astype(np.float32)

    return id_encoded_mask, surviving_ids, surviving_scores


def _relabel_id_encoded_mask_to_dense(
    id_encoded_mask,
):
    """Relabel old IDs -> dense 1..M.

    This is so they fit the scores width and have no gaps.
    """
    # get old mask ids
    old_mask_ids = np.asarray(
        [id for id in np.unique(id_encoded_mask) if id != 0]
    )
    n_old_mask_ids = len(old_mask_ids)

    # map old (array index) to new (array value) IDs
    old_to_new_ids = np.zeros(id_encoded_mask.max() + 1, dtype=np.int16)
    old_to_new_ids[old_mask_ids] = np.arange(
        1, n_old_mask_ids + 1, dtype=np.int16
    )

    # apply map to id_encoded_mask
    new_id_encoded_mask = old_to_new_ids[
        id_encoded_mask
    ]  # background (0) -> 0
    new_ids = np.arange(1, n_old_mask_ids + 1)

    return new_id_encoded_mask, new_ids


def _filter_instances_by_area_solidity(
    id_encoded_mask, surviving_ids, surviving_scores, max_area, min_solidity
):
    """Drop merged instances above ``max_area`` or below ``min_solidity``.

    Operates on the assembled instances (after the overlap merge) rather than
    on per-tile regions, so a burrow clipped at a tile seam is judged as one
    whole blob instead of as its smaller, more-convex fragments. This is the
    counterpart of the per-tile ``min_area`` pre-filter in
    ``_filter_masks_into_regions``; see its docstring for the rationale.

    ``surviving_ids`` are the instance IDs present in ``id_encoded_mask``
    (ascending) and ``surviving_scores`` is index-aligned to them. Surviving
    instances are relabelled to dense ``1..M``. Returns ``(new_id_encoded_mask,
    new_ids, new_scores, drop_counts)`` with ``new_ids`` ascending,
    ``new_scores`` index-aligned, and ``drop_counts`` counting the instances
    dropped per reason.
    """
    drop_counts = {"area_high": 0, "solidity": 0}
    score_by_id = dict(
        zip(surviving_ids.tolist(), surviving_scores.tolist(), strict=True)
    )

    # Keep instances within the area cap and above the solidity threshold.
    # (solidity is the ratio of region pixels to convex-hull pixels, in [0, 1],
    # a proxy for blob-likeness)
    keep_ids = []
    for region in regionprops(id_encoded_mask):
        if region.area > max_area:
            drop_counts["area_high"] += 1
            continue
        if region.solidity < min_solidity:
            drop_counts["solidity"] += 1
            continue
        keep_ids.append(region.label)

    # Handle the all-dropped case
    if not keep_ids:
        return (
            np.zeros_like(id_encoded_mask),
            np.array([], dtype=np.int16),
            np.array([], dtype=np.float32),
            drop_counts,
        )

    # Zero out dropped instances and relabel survivors to dense 1..M
    id_encoded_mask = id_encoded_mask.copy()
    id_encoded_mask[~np.isin(id_encoded_mask, keep_ids)] = 0
    new_id_encoded_mask, new_ids = _relabel_id_encoded_mask_to_dense(
        id_encoded_mask
    )

    # Align scores to the ascending kept IDs (relabel preserves their order)
    new_scores = np.array(
        [score_by_id[i] for i in sorted(keep_ids)], dtype=np.float32
    )

    return new_id_encoded_mask, new_ids, new_scores, drop_counts


def _cap_regions_per_image(
    id_encoded_mask, surviving_ids, surviving_scores, max_regions
):
    """Cap the number of regions per frame to the top ``max_regions`` scoring.

    Keeps only the highest-scoring ``max_regions`` instances, drops the rest
    from ``id_encoded_mask``, and relabels the remaining IDs to dense ``1..M``
    (capping leaves gaps). Returns ``(new_id_encoded_mask, new_ids,
    new_scores)`` with ``new_ids`` ascending and ``new_scores`` index-aligned.

    If there are no more than ``max_regions`` regions, inputs are returned
    unchanged (the merge already yields dense labels).
    """
    if len(surviving_ids) <= max_regions:
        return id_encoded_mask, surviving_ids, surviving_scores

    # we select the top M scoring ones
    capped_sorted_idcs = np.argsort(surviving_scores)[::-1][:max_regions]
    # Get corresponding "M" IDs and scores in score-order
    selected_ids = surviving_ids[capped_sorted_idcs]
    selected_scores = surviving_scores[capped_sorted_idcs]

    # Drop non-selected IDs from the mask
    id_encoded_mask[~np.isin(id_encoded_mask, selected_ids)] = 0

    # Re-sort the scores into ascending ID order from the selected
    # IDs; this is required for the relabel step
    order = np.argsort(selected_ids)
    new_scores = selected_scores[order]

    # Capping leaves gaps in the IDs, so relabel old IDs -> dense
    # 1..M (without capping the merge already yields dense labels)
    new_id_encoded_mask, new_ids = _relabel_id_encoded_mask_to_dense(
        id_encoded_mask
    )

    return new_id_encoded_mask, new_ids, new_scores


def _toggle_button(label, trace_idcs):
    """Build a Plotly restyle button that toggles the given traces' visibility.

    Mirrors the helper in notebook_burrows_sam3_pass_2.py: the primary action
    (``args``) hides the traces and the alternate action (``args2``) shows
    them, so each click flips their visibility.
    """
    if isinstance(trace_idcs, int):
        trace_idcs = [trace_idcs]
    return dict(
        label=label,
        method="restyle",
        args=[{"visible": False}, list(trace_idcs)],
        args2=[{"visible": True}, list(trace_idcs)],
    )


def _get_video_length(ds_video):
    """Return a video's length as ``(minutes, n_frames)`` from its zarr coords.

    Copied from notebook_burrows_sam3_pass_2.py. A clip spans from the end of
    the previous escape (or the video start) to the end of the current escape.
    """
    n_frames = int(ds_video.clip_last_frame_0idx.max().compute()) + 1
    return (n_frames / float(ds_video.fps) / 60, n_frames)


def _compute_rasterised_trajectories_array(
    dt, video_str, canvas, img_h_i, img_w_i, traj_color, dynspread_th
):
    """Return (H, W, 4) uint8 RGBA with the video's trajectories.

    Copied from annotate_burrow_prompts_manual.py. Returns a fully transparent
    canvas when the video is absent from the datatree or has no valid
    coordinates.
    """
    import datashader.transfer_functions as tf

    # Return zeros if video not in dataset
    if video_str not in dt:
        return np.zeros((img_h_i, img_w_i, 4), dtype=np.uint8)

    # Get non-nan x,y coords
    position = dt[video_str].to_dataset().position
    x = position.sel(space="x").values.reshape(-1)
    y = position.sel(space="y").values.reshape(-1)
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Return zeros if no valid coords
    if not valid.any():
        return np.zeros((img_h_i, img_w_i, 4), dtype=np.uint8)

    # add data to canvas and rasterise
    agg = canvas.points(pd.DataFrame({"x": x[valid], "y": y[valid]}), "x", "y")
    shaded = tf.shade(agg, cmap=[traj_color])
    shaded = tf.dynspread(shaded, threshold=dynspread_th)

    # datashader y-origin is bottom; flip to match image y-origin (top)
    return np.array(shaded.to_pil().transpose(Image.FLIP_TOP_BOTTOM))


def _write_html_plots(  # noqa: C901
    image_array,
    list_video_per_img,
    points_xy_px_per_video,
    root,
    output_dir,
    masks_zarr_stem,
    trajectories_zarr=None,
    min_hits_per_burrow_frac=0.10,
    dynspread_threshold=0.975,
):
    """Write one interactive Plotly HTML plot per frame.

    Each plot overlays the colored mask instances and the manual point prompts
    on the frame, plus invisible hover markers at every mask centroid carrying
    the mask ID and its SAM3 score. Every layer is individually toggleable.

    When ``trajectories_zarr`` is given (a CrabTracks xarray datatree keyed by
    video), each plot additionally gets a datashader trajectory-raster layer
    and red contours around burrows occupied by a crab in at least
    ``min_hits_per_burrow_frac`` of the video's frames; the title is enriched
    with the video length and hit count. These trajectory layers reproduce
    notebook_burrows_sam3_pass_2.py.

    Masks and scores are read back from the just-written per-video zarr groups
    (``root[f"{video}/masks"]`` is ``(H, W)`` and ``root[f"{video}/scores"]``
    is ``(max_regions + 1,)``, indexed by mask ID); frames are read lazily from
    ``image_array``. One HTML is written per frame, including frames with no
    prompts or no masks.
    """
    import plotly.graph_objects as go

    img_h, img_w = image_array.img_h, image_array.img_w

    # Open the optional trajectories datatree and build a reusable canvas
    # (frames all share the same size, so one canvas serves every frame).
    dt = None
    canvas = None
    if trajectories_zarr is not None:
        import datashader as ds
        import xarray as xr

        dt = xr.open_datatree(trajectories_zarr, engine="zarr", chunks={})
        canvas = ds.Canvas(
            plot_width=img_w,
            plot_height=img_h,
            x_range=(0, img_w),
            y_range=(0, img_h),
        )

    # Create a single timestamped output directory for this run's plots
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_plots = output_dir / f"plots_{masks_zarr_stem}_{plot_timestamp}"
    output_dir_plots.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(len(image_array)):
        video_str = list_video_per_img[frame_idx]
        # Per-video arrays: masks is (H, W); scores is (max_regions + 1,)
        # indexed by mask ID (column 0 = background, unused/NaN).
        id_mask = root[f"{video_str}/masks"][:].astype(np.int32)
        scores_row = root[f"{video_str}/scores"][:]
        n_masks = len(np.unique(id_mask)) - (1 if (id_mask == 0).any() else 0)

        # Colored mask overlay (tab10 cycled over IDs, ~0.5 alpha)
        mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        nonzero = id_mask > 0
        mask_rgba[nonzero, :3] = TAB10_RGB[(id_mask[nonzero] - 1) % 10]
        mask_rgba[nonzero, 3] = 128  # fill alpha ~= 0.5

        # Build the figure with the frame as the background image
        fig = go.Figure()
        fig.add_layout_image(
            source=Image.fromarray(image_array[frame_idx]),
            xref="x",
            yref="y",
            x=0,
            y=0,  # top-left corner in data coords (y is inverted)
            sizex=img_w,
            sizey=img_h,
            sizing="stretch",
            layer="below",
        )

        # Mask fills layer
        masks_trace_idx = len(fig.data)
        fig.add_trace(
            go.Image(
                z=mask_rgba,
                colormodel="rgba256",
                name=f"masks ({n_masks})",
                hoverinfo="skip",
            )
        )

        # Optional trajectory-derived layers (contours + raster)
        contour_trace_idcs = []
        traj_trace_idx = None
        n_hit_ids = 0
        video_minutes = None
        if dt is not None:
            # Per-burrow trajectory hit counts -> high-activity burrow IDs
            hit_ids = np.array([], dtype=np.int64)
            if video_str in dt:
                video_minutes, video_n_frames = _get_video_length(
                    dt[video_str].to_dataset()
                )
                position = dt[video_str].to_dataset().position
                x = position.sel(space="x").values.reshape(-1)
                y = position.sel(space="y").values.reshape(-1)
                valid = ~np.isnan(x) & ~np.isnan(y)
                cols = np.round(x[valid]).astype(int)
                rows = np.round(y[valid]).astype(int)
                in_frame = (
                    (cols >= 0) & (cols < img_w) & (rows >= 0) & (rows < img_h)
                )
                hits_per_id = np.bincount(
                    id_mask[rows[in_frame], cols[in_frame]],
                    minlength=int(id_mask.max()) + 1,
                )
                min_hits = int(min_hits_per_burrow_frac * video_n_frames)
                hit_ids = np.where(hits_per_id[1:] >= min_hits)[0] + 1
            n_hit_ids = len(hit_ids)

            # Red contours around high-activity burrows
            for mid in hit_ids:
                for contour in find_contours(id_mask == mid, 0.5):
                    contour_trace_idcs.append(len(fig.data))
                    fig.add_trace(
                        go.Scatter(
                            x=contour[:, 1],
                            y=contour[:, 0],
                            mode="lines",
                            line=dict(color="red", width=1.5),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            # Trajectory raster layer
            traj_rgba = _compute_rasterised_trajectories_array(
                dt,
                video_str,
                canvas,
                img_h,
                img_w,
                "#c3ff1f",
                dynspread_threshold,
            )
            traj_trace_idx = len(fig.data)
            fig.add_trace(
                go.Image(
                    z=traj_rgba,
                    colormodel="rgba256",
                    name="trajectories",
                    hoverinfo="skip",
                )
            )

        # Manual point prompts (already pixel coords in this script)
        prompts_trace_idx = None
        manual_pts = points_xy_px_per_video.get(video_str)
        if manual_pts is not None and len(manual_pts) > 0:
            prompts_trace_idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=manual_pts[:, 0],
                    y=manual_pts[:, 1],
                    mode="markers",
                    marker=dict(symbol="x", color="lime", size=10),
                    name=f"manual prompts ({len(manual_pts)})",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Invisible hover markers at each mask centroid (ID + SAM3 score)
        scores_trace_idx = None
        regions = regionprops(id_mask)
        if regions:
            centroid_ids = np.array([r.label for r in regions])
            centroids = np.array(
                [r.centroid for r in regions]
            )  # (n, row, col)
            centroid_scores = scores_row[centroid_ids]
            scores_trace_idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 1],
                    y=centroids[:, 0],
                    mode="markers",
                    marker=dict(size=12, color="rgba(0,0,0,0)"),
                    customdata=np.stack(
                        [centroid_ids, centroid_scores], axis=1
                    ),
                    name="scores",
                    showlegend=False,
                    hovertemplate=(
                        "id=%{customdata[0]:.0f}<br>"
                        "score=%{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )

        # Title and toggle buttons
        frame_stem = image_array.img_paths[frame_idx].stem
        if video_minutes is not None:
            title = (
                f"{frame_stem} - ({video_minutes:.1f} min) {n_masks} masks "
                f"({n_hit_ids} with >= "
                f"{min_hits_per_burrow_frac * 100:.0f}% frames with crab)"
            )
        else:
            title = f"{frame_stem} - {n_masks} masks"

        buttons = [_toggle_button("toggle masks", masks_trace_idx)]
        if prompts_trace_idx is not None:
            buttons.append(_toggle_button("toggle prompts", prompts_trace_idx))
        if scores_trace_idx is not None:
            buttons.append(_toggle_button("toggle scores", scores_trace_idx))
        if contour_trace_idcs:
            buttons.append(
                _toggle_button("toggle contours", contour_trace_idcs)
            )
        if traj_trace_idx is not None:
            buttons.append(
                _toggle_button("toggle trajectories", traj_trace_idx)
            )

        fig.update_layout(
            title=title,
            xaxis_title="x (pixels)",
            yaxis_title="y (pixels)",
            yaxis_scaleanchor="x",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    xanchor="left",
                    y=1.08,
                    yanchor="bottom",
                    showactive=False,
                    buttons=buttons,
                )
            ],
            xaxis=dict(
                range=[0, img_w],
                showgrid=False,
                zeroline=False,
                linecolor="black",
                mirror=True,
                ticks="outside",
            ),
            yaxis=dict(
                range=[img_h, 0],  # invert y so image origin is top-left
                showgrid=False,
                zeroline=False,
                linecolor="black",
                mirror=True,
                ticks="outside",
            ),
        )

        output_html = output_dir_plots / f"masks_{frame_stem}.html"
        fig.write_html(
            str(output_html),
            include_plotlyjs=True,
            config={"toImageButtonOptions": {"format": "svg"}},
        )

    print(f"Saved {len(image_array)} HTML plots to {output_dir_plots}")


def main(args: argparse.Namespace) -> None:  # noqa: C901
    """Run SAM3 burrow segmentation per frame and write masks to zarr."""
    # ------------------------------------------------------------------
    # Load frames as a lazy array and map each frame to its video
    list_image_files = sorted(Path(args.images_dir).glob("*.png"))
    image_array = ImageArrayLazy(list_image_files)
    print(f"Loaded {len(image_array)} frames of shape {image_array.shape[1:]}")

    # we assume one video per image only
    list_video_per_img = [
        img_p.stem.split("_", 1)[0].split("-Loop")[0]
        for img_p in list_image_files
    ]

    # ------------------------------------------------------------------
    # Resolve tile geometry. Defaults derive from the image height (a third of
    # it for the tile side, half of that for the overlap)
    image_shape = image_array.shape[:3]
    n_images, image_h, image_w = image_shape
    tile_size = args.tile_size if args.tile_size is not None else image_h // 3
    tile_overlap = (
        args.tile_overlap if args.tile_overlap is not None else tile_size // 2
    )

    # Compute tile (x0,y0,x1,y1) coordinates
    tiles = _make_tiles(image_h, image_w, tile_size, tile_overlap)
    print(
        f"Tiling each {image_h}x{image_w} frame into {len(tiles)} tiles "
        f"(size {tile_size}px, overlap {tile_overlap}px)"
    )

    # ------------------------------------------------------------------
    # Initialise the output ID-encoded mask zarr store (includes scores)
    metadata_dict = {
        "sam3_model": "sam3_image",
        "source_images_dir": str(args.images_dir),
        "manual_prompts_csv": str(args.manual_prompts_csv),
        "n_images": n_images,
        "image_shape": [image_h, image_w],
        "estim_max_regions_per_image": args.max_regions_per_image,
        "text_prompt": args.text_prompt,
        "sam3_confidence_threshold": args.conf_threshold,
        "inference_mode": "tiled",
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "postproc_min_mask_area_PIXELS": args.min_mask_area_pixels,
        "postproc_max_mask_area_PIXELS": args.max_mask_area_pixels,
        "postproc_min_solidity": args.min_solidity,
        "postproc_overlap_threshold": args.overlap_threshold,
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_first_index": 1,
        # mask instance IDs in the zarr store start at 1 (not 0),
        # because 0 is reserved for the background label.
    }
    root, output_masks_zarr = _initialise_mask_zarr(
        Path(args.output_dir),
        metadata_dict,
    )

    # ------------------------------------------------------------------
    # Extract point prompts per video (pixel coords, keyed by video)
    points_xy_px_per_video = _extract_point_prompts_per_video(
        args.manual_prompts_csv
    )

    # ------------------------------------------------------------------
    # Build SAM3 image model and processor under inference/autocast contexts.
    # autocast avoids a mismatch between bfloat16 (model params) and float
    # (input); it is only enabled when running on CUDA.
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        # Instantiate model (holds weights and submodules: backbone, text
        # encoder, grounding head, mask decoder) and the processor (a wrapper
        # handling I/O conversion, inference_state mutation, prompt
        # accumulation and confidence thresholding; model is at
        # ``processor.model``).
        model = build_sam3_image_model()
        processor = Sam3Processor(
            model, confidence_threshold=args.conf_threshold
        )

        # Encode the text prompt once and reuse it across every tile and frame.
        text_features = (
            model.backbone.forward_text(
                [args.text_prompt],
                device=processor.device,
            )
            if args.text_prompt is not None
            else None
        )

        # --------------------------------------------------------------
        # Run inference on every frame and write ID-encoded masks to zarr
        count_postproc_frames_empty = 0
        postproc_frames_w_masks = []
        # (frame_idx, n_masks) for every frame run through inference, including
        # empty ones (0 masks); excludes frames skipped for having no prompts.
        n_masks_per_frame = []

        for frame_idx in range(len(image_array)):
            # Initialise zarr arrays
            # ID-encoded mask array
            video_str = list_video_per_img[frame_idx]
            root.create_array(
                f"{video_str}/masks",
                shape=(image_h, image_w),
                dtype="int16",
                fill_value=0,  # background
                chunks=(image_h, image_w),
            )

            # Initialise scores array as a sibling of "masks", indexed by mask
            # ID. Shape (max_regions + 1,); index 0 = background (unused, NaN).
            root.create_array(
                f"{video_str}/scores",
                shape=(args.max_regions_per_image + 1,),
                chunks=(args.max_regions_per_image + 1,),
                dtype="float32",
                fill_value=np.nan,
            )

            # -----------------------------------------
            # Get point prompts (full-image pixel x, y) for the matching video
            prompts_img_px = points_xy_px_per_video.get(video_str)
            if prompts_img_px is None or len(prompts_img_px) == 0:
                print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
                continue

            # Load image as a numpy array (H, W, C) so tiles can be cropped
            img_full = image_array[frame_idx]

            # -------------------------------------------------------------
            # Run SAM3 tile by tile and pool the filtered regions of every
            # prompted tile (mapped back onto the full-image canvas).
            list_tile_masks = []
            list_tile_scores = []
            tile_drop_counts = {"area_low": 0}
            n_tiles_used = 0
            for x0, y0, x1, y1 in tiles:
                # Select prompts inside this tile (full-image pixel coords)
                in_tile = (
                    (prompts_img_px[:, 0] >= x0)
                    & (prompts_img_px[:, 1] >= y0)
                    & (prompts_img_px[:, 0] < x1)
                    & (prompts_img_px[:, 1] < y1)
                )
                # Skip tiles with no prompt point
                if not in_tile.any():
                    continue
                n_tiles_used += 1

                # Express in-tile prompts as crop-local pixels, then normalise
                # to the crop size for SAM3
                prompts_crop_px = prompts_img_px[in_tile] - np.array(
                    [x0, y0],
                    dtype=np.float32,
                )
                prompts_crop_norm = prompts_crop_px / np.array(
                    [x1 - x0, y1 - y0],
                    dtype=np.float32,
                )

                # Add prompts (text + points) to the inference state per crop
                crop_pil = Image.fromarray(img_full[y0:y1, x0:x1])
                inference_state = _add_prompts_to_inference_state(
                    processor,
                    crop_pil,
                    prompts_crop_norm,
                    text_features=text_features,
                )

                # Get predicted masks and scores, then release GPU state
                masks, scores, n_objects = _extract_sam3_results_one_img(
                    inference_state
                )
                del inference_state

                if n_objects == 0:
                    continue

                # Split tile masks into regions and drop sub-min-area speckle
                # (all in crop coords; max-area and solidity are applied later,
                # on the merged instances)
                kept_masks, kept_scores, drop_counts = (
                    _filter_masks_into_regions(
                        masks,
                        scores,
                        args.min_mask_area_pixels,
                    )
                )
                for reason, count in drop_counts.items():
                    tile_drop_counts[reason] += count

                # Map each kept region into a full-image boolean canvas
                for crop_mask in kept_masks:
                    canvas = np.zeros((image_h, image_w), dtype=bool)
                    canvas[y0:y1, x0:x1] = crop_mask
                    list_tile_masks.append(canvas)
                list_tile_scores.extend(kept_scores.tolist())

            # -------------------------------------------------------------
            # Merge the pooled regions across tiles into instances by overlap
            n_regions_pre_merge = len(list_tile_masks)
            id_encoded_mask, merged_ids, merged_scores = (
                _merge_bool_masks_by_overlap(
                    list_tile_masks,
                    np.asarray(list_tile_scores, dtype=np.float32),
                    (image_h, image_w),
                    args.overlap_threshold,
                )
            )
            n_merged_regions = len(merged_ids)

            # Filter the assembled instances by max-area and solidity (applied
            # here, after the merge, so a burrow clipped at a tile seam is
            # judged as one whole blob rather than as its more-convex pieces)
            (
                id_encoded_mask,
                surviving_ids,
                surviving_scores,
                instance_drops,
            ) = _filter_instances_by_area_solidity(
                id_encoded_mask,
                merged_ids,
                merged_scores,
                args.max_mask_area_pixels,
                args.min_solidity,
            )

            # -------------------------
            # Log postprocessing results for this frame
            n_surviving_regions = len(surviving_ids)
            if n_surviving_regions == 0:
                print(
                    f"Frame {frame_idx} ({video_str}): "
                    f"no masks after postprocessing "
                    f"({n_tiles_used}/{len(tiles)} prompted tiles)"
                )
                count_postproc_frames_empty += 1
                n_masks_per_frame.append((frame_idx, 0))
                continue

            n_detected_regions = (
                n_regions_pre_merge + tile_drop_counts["area_low"]
            )
            print(
                f"Frame {frame_idx} ({video_str}): "
                f"{n_detected_regions} regions detected; "
                f"after min-area filter: {n_regions_pre_merge}; "
                f"after overlap merge: {n_merged_regions}; "
                f"after area/solidity filter: {n_surviving_regions} "
                f"(dropped {instance_drops['area_high']} large, "
                f"{instance_drops['solidity']} non-solid); "
                f"before capping; "
                f"{n_tiles_used}/{len(tiles)} prompted tiles.",
            )
            # -------------------------

            # Enforce the per-frame cap on max number of regions: keep only
            # the top-scoring ones and relabel the surviving IDs to dense 1..M
            new_id_encoded_mask, new_ids, surviving_scores = (
                _cap_regions_per_image(
                    id_encoded_mask,
                    surviving_ids,
                    surviving_scores,
                    args.max_regions_per_image,
                )
            )

            # Save results to zarr
            root[f"{video_str}/masks"] = new_id_encoded_mask
            root[f"{video_str}/scores"][new_ids] = surviving_scores

            # -------------------------
            # Log final number of masks
            n_masks_saved = len(new_ids)
            n_masks_per_frame.append((frame_idx, n_masks_saved))
            print(f"Frame {frame_idx} ({video_str}): {n_masks_saved} masks")

            postproc_frames_w_masks.append(frame_idx)

    # Add extra metrics to zarr array
    root.attrs["frames_with_masks"] = postproc_frames_w_masks
    root.attrs["n_masks_per_frame"] = n_masks_per_frame
    print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")

    # ----------------------------
    # Summarise masks-per-frame statistics over every frame run through
    # inference, including empty frames (0 masks).
    print(
        f"N frames with no masks after postprocessing: "
        f"{count_postproc_frames_empty}"
    )
    if n_masks_per_frame:
        counts = np.array([n_masks for _, n_masks in n_masks_per_frame])
        mean_masks = counts.mean()
        print(
            f"Stats for masks per frame (n={len(counts)} frames)"
            f"mean={mean_masks:.2f}, "
            f"median={np.median(counts):.1f}, "
            f"min={counts.min()}, max={counts.max()}"
        )

        # Frame indices with fewer than the mean number of masks per frame
        frames_below_mean = [
            frame_idx
            for frame_idx, n_masks in n_masks_per_frame
            if n_masks < mean_masks
        ]
        print(
            f"Frames with fewer than mean ({mean_masks:.2f}) masks per frame "
            f"({len(frames_below_mean)} frames): {frames_below_mean}"
        )

    # ----------------------------
    # Optionally write one interactive HTML plot per frame (masks + prompts +
    # scores, plus trajectory layers when a trajectories zarr is given).
    if args.save_html_plots:
        _write_html_plots(
            image_array,
            list_video_per_img,
            points_xy_px_per_video,
            root,
            Path(args.output_dir),
            output_masks_zarr.stem,
            trajectories_zarr=args.trajectories_zarr,
            min_hits_per_burrow_frac=args.min_hits_per_burrow_frac,
        )


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Segment burrows with SAM3 from per-video point prompts and "
            "write the postprocessed, ID-encoded masks to a zarr store."
        ),
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory of PNG frames to run inference on.",
    )
    parser.add_argument(
        "manual_prompts_csv",
        type=Path,
        help=(
            "CSV of per-video manual point prompts, as produced by "
            "annotate_burrow_prompts_manual.py (columns: group_id, "
            "prompt_point_x, prompt_point_y)."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Output directory. A timestamped "
            "'masks_<YYYYMMDD_HHMMSS>.zarr' "
            "store is created inside it, so multiple runs never collide."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="hole",
        help=(
            "Text prompt passed to SAM3 alongside the point prompts "
            "(default: 'hole'). Pass an empty string to disable the text "
            "prompt and rely on the point prompts only."
        ),
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help=(
            "SAM3 confidence threshold; masks below this score are not scaled "
            "up to full resolution (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--min-mask-area-pixels",
        type=int,
        default=200,
        help=(
            "Connected regions smaller than this area (in pixels) are dropped "
            "during postprocessing (default: 200)."
        ),
    )
    parser.add_argument(
        "--max-mask-area-pixels",
        type=int,
        default=2500,
        help=(
            "Connected regions larger than this area (in pixels) are dropped "
            "during postprocessing (default: 2500)."
        ),
    )
    parser.add_argument(
        "--min-solidity",
        type=float,
        default=0.95,
        help=(
            "Minimum region solidity (region area / convex hull area, in "
            "[0, 1]) kept during postprocessing (default: 0.95)."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Side length in pixels of the (square) tiles SAM3 runs on. "
            "Defaults to a third of the image height when omitted."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help=(
            "Overlap in pixels between neighbouring tiles. Should exceed a "
            "burrow's diameter so each burrow is fully contained in at least "
            "one tile. Defaults to half the tile size when omitted."
        ),
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help=(
            "Fraction of the smaller region's area that the intersection of "
            "two regions must cover for them to be merged into one instance "
            "(intersection / min(area_i, area_j), in [0, 1]). Merging is "
            "transitive and each instance inherits the max contributor score "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--max-regions-per-image",
        type=int,
        default=500,
        help=(
            "Estimated upper bound on kept regions per frame; sets the number "
            "of columns of the scores array (default: 500)."
        ),
    )
    parser.add_argument(
        "--save-html-plots",
        action="store_true",
        help=(
            "If set, also write one interactive Plotly HTML plot per frame "
            "into output_dir, overlaying the colored mask instances and "
            "manual point prompts on the frame (masks/prompts/scores "
            "toggleable, with the mask ID and SAM3 score on hover). "
            "Default: not set."
        ),
    )
    parser.add_argument(
        "--trajectories-zarr",
        type=Path,
        default=None,
        help=(
            "Optional CrabTracks trajectories zarr (an xarray datatree keyed "
            "by video). When given alongside --save-html-plots, each plot "
            "also gets a datashader trajectory-raster layer and red activity "
            "contours for burrows hit by a crab in at least "
            "--min-hits-per-burrow-frac of frames."
        ),
    )
    parser.add_argument(
        "--min-hits-per-burrow-frac",
        type=float,
        default=0.10,
        help=(
            "Fraction of a video's frames a crab must occupy a burrow for it "
            "to get a red activity contour in the HTML plots (only used with "
            "--trajectories-zarr; default: 0.10)."
        ),
    )

    args = parser.parse_args(list_args)

    # An empty text prompt disables the text prompt entirely.
    if args.text_prompt == "":
        args.text_prompt = None

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
