"""Interactive SAM3 burrow segmentation in a napari viewer.

A prototyping notebook that turns the batch workflow of
``notebook_burrows_sam3.py`` into an interactive tool. The interactive logic
lives in a single self-contained ``Sam3BurrowWidget`` (a ``QWidget``), which is
attached to a napari viewer in the last cell.

  * Load an array of images into napari (one Image layer, navigated with the
    dims slider).
  * Click point prompts directly on each image (one nD Points layer, where the
    first coordinate is the frame index).
  * Press "Run inference" to run SAM3 on the image currently in view, using all
    point prompts placed on that frame. Predicted masks appear in a Labels
    layer.
  * Tune the confidence threshold *post-hoc* with a slider: SAM3 runs once at a
    low "inference floor"; the slider just re-filters the cached masks, so it is
    instant and does not touch the GPU.
  * Refine by adding/removing points and pressing "Run" again.
  * Manually edit masks with napari's paintbrush; edits auto-save to the zarr
    store (on every paint stroke and whenever you navigate to another frame).

Reproducibility: the point prompts per frame, the confidence threshold per
frame, the image file list and the SAM3 commit are all written to the zarr
store metadata. ``Sam3BurrowWidget.load_prompts_from_zarr`` reloads the prompts
from a store.

Why an "inference floor": setting the SAM3 ``confidence_threshold`` very low
makes the model return a huge number of masks and causes CUDA OOM. Instead we
run once at a moderate floor (default 0.2) and keep every mask above it with its
score; the post-hoc slider can only move *up* from the floor. To explore lower
confidences, lower the floor in the UI and re-run.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "sam3 @ git+https://github.com/facebookresearch/sam3.git",
#   "einops",
#   "huggingface_hub",
#   "scikit-image",
#   "scipy",
#   "napari[all]",
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
import os

# Reduce CUDA allocator fragmentation. Must be set before torch is
# imported -> restart the kernel for this to take effect in a notebook.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import importlib.metadata
import json
import logging
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import torch
import zarr
from PIL import Image
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy import ndimage as ndi
from superqt import QLabeledDoubleSlider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("napari_sam3")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data and defaults

# Directory of PNG frames to load into the viewer.
images_dir = "/home/sminano/swc/project_crabs/burrow_mean_image_slurm_3014447"

# Output dir for the masks zarr store.
OUTPUT_DIR = Path(
    "/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3"
)

# Defaults for the UI controls (all editable in the widget once it is open).
DEFAULT_TEXT_PROMPT = "burrow"  # empty string -> geometric-only inference
DEFAULT_INFERENCE_FLOOR = 0.2  # SAM3 confidence_threshold used at inference
DEFAULT_MAX_MASK_FRAC = 0.10  # drop masks larger than this fraction of pixels

# TODO: change to number of pixels
DEFAULT_MIN_MASK_FRAC = (
    0.000001  # drop masks smaller than this fraction of pixels
)


# %%%%%%%%%%
# Helpers


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def create_mask_zarr(path_to_zarr, zarr_array_shape, metadata_dict=None):
    """Create a zarr store for ID-encoded masks and write metadata."""
    n_images, image_h, image_w = zarr_array_shape[:3]
    mask_zarr = zarr.open(
        path_to_zarr,
        mode="w",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,
        chunks=(1, image_h, image_w),
    )
    if metadata_dict is not None:
        mask_zarr.attrs.update(metadata_dict)
    return mask_zarr


def add_point_prompt(processor, state, point_xy, label=True):
    """Add a single point prompt and run inference, returning the updated state.

    ``point_xy`` is an ``(x, y)`` pair normalized to ``[0, 1]``. Mirrors
    ``Sam3Processor.add_geometric_prompt`` but appends a point to the geometric
    prompt instead of a box. Relies on SAM3 internals (``_get_dummy_prompt`` /
    ``_forward_grounding``) as the processor exposes no public point method.
    """
    if "backbone_out" not in state:
        raise ValueError("call processor.set_image before adding a prompt")
    if "language_features" not in state["backbone_out"]:
        # no text prompt yet: fall back to a dummy "visual" text prompt so the
        # model relies only on the geometric prompt
        dummy_text = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    # points: (n_points, batch, 2); labels: (n_points, batch); mask: (batch, n)
    pts = torch.tensor(
        point_xy, device=processor.device, dtype=torch.float32
    ).view(1, 1, 2)
    lbl = torch.tensor(
        [label], device=processor.device, dtype=torch.bool
    ).view(1, 1)
    msk = torch.zeros(1, 1, dtype=torch.bool, device=processor.device)
    state["geometric_prompt"].append_points(pts, lbl, msk)

    return processor._forward_grounding(state)


def get_sam3_commit():
    """Best-effort SAM3 commit hash (for reproducibility metadata)."""
    try:
        dist = importlib.metadata.distribution("sam3")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            info = json.loads(direct_url)
            vcs = info.get("vcs_info", {})
            return vcs.get("commit_id") or info.get("url", "unknown")
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    try:
        return importlib.metadata.version("sam3")
    except Exception:  # noqa: BLE001
        return "unknown"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Postprocessing


def postprocess(id_mask, frame_idx, max_mask_frac, min_mask_frac):
    """Clean an ID-encoded mask; returns the cleaned mask and logs warnings.

    Order matters: disconnected masks are split first, so a small fragment
    split off a larger mask is still size-checked afterwards.

    - Disconnected masks: a mask whose pixels form several separate blobs is
      split, each connected blob becoming its own fresh ID.
    - Oversized masks: an ID covering more than ``max_mask_frac`` of the image
      is dropped.
    - Undersized masks: an ID covering less than ``min_mask_frac`` is dropped.
    - Overlap: already resolved upstream by the max-ID encoding.
    """
    total_px = id_mask.shape[0] * id_mask.shape[1]
    result = np.zeros_like(id_mask)
    next_id = 1

    for orig_id in np.unique(id_mask):
        if orig_id == 0:
            continue
        binary = id_mask == orig_id

        # number each connected blob of this ID separately
        labelled, n_blobs = ndi.label(binary)
        new_ids = []
        for blob in range(1, n_blobs + 1):
            blob_mask = labelled == blob
            frac = blob_mask.sum() / total_px
            if frac > max_mask_frac:
                logger.warning(
                    "frame %d, ID %d: dropped oversized mask "
                    "(%.1f%% of image > %.1f%%)",
                    frame_idx,
                    orig_id,
                    frac * 100,
                    max_mask_frac * 100,
                )
                continue
            if frac < min_mask_frac:
                logger.warning(
                    "frame %d, ID %d: dropped undersized mask "
                    "(%.3f%% of image < %.3f%%)",
                    frame_idx,
                    orig_id,
                    frac * 100,
                    min_mask_frac * 100,
                )
                continue
            result[blob_mask] = next_id
            new_ids.append(next_id)
            next_id += 1

        if n_blobs > 1 and new_ids:
            logger.warning(
                "frame %d, ID %d: mask was disconnected (%d blobs) -> "
                "split into IDs %s",
                frame_idx,
                orig_id,
                n_blobs,
                new_ids,
            )

    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# The widget


class Sam3BurrowWidget(QWidget):
    """Self-contained widget for interactive SAM3 burrow segmentation.

    On construction it creates the napari layers (Image, Points, Labels), a
    zarr store for the masks, and the UI controls. Attach it to a viewer with
    ``viewer.window.add_dock_widget(widget)``.
    """

    def __init__(
        self,
        napari_viewer,
        image_array,
        model,
        output_dir,
        parent=None,
    ):
        """Build layers, the zarr store and the UI for ``napari_viewer``."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.image_array = image_array
        self.model = model
        self.n_images, self.image_h, self.image_w = image_array.shape[:3]

        # Per-frame state.
        # - inference_cache: raw SAM3 output per frame, so the threshold slider
        #   can re-filter instantly without re-running the model.
        # - point_prompts_per_frame, threshold_per_frame: reproducibility.
        self.inference_cache: dict[int, dict] = {}
        self.point_prompts_per_frame: dict[int, list] = {}
        self.threshold_per_frame: dict[int, float] = {}
        self.annotated_frames: set[int] = set()

        self._create_mask_store(output_dir)
        self._create_layers()

        self.setLayout(QFormLayout())
        self._create_text_prompt_widget()
        self._create_inference_floor_widget()
        self._create_max_mask_frac_widget()
        self._create_min_mask_frac_widget()
        self._create_run_button()
        self._create_threshold_widget()

        self._connect_layer_events()

    # ------------------------------------------------------------------
    # Construction helpers

    def _create_mask_store(self, output_dir):
        """Create the output ID-encoded mask zarr store and in-memory copy."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_masks_zarr = output_dir / f"masks_napari_{timestamp}.zarr"
        self.sam3_commit = get_sam3_commit()

        self.mask_zarr = create_mask_zarr(
            self.output_masks_zarr,
            (self.n_images, self.image_h, self.image_w),
            metadata_dict={
                "timestamp": timestamp,
                "sam3_model": "sam3_image",
                "sam3_commit": self.sam3_commit,
                "source_images_dir": str(
                    self.image_array.img_paths[0].parent
                ),
                "image_shape": [self.image_h, self.image_w],
                "image_files": [
                    str(p) for p in self.image_array.img_paths
                ],
                "mask_encoding": "instance_id",
                "background_label": 0,
                "id_offset": 1,
            },
        )
        logger.info("Created mask zarr store at %s", self.output_masks_zarr)

        # In-memory copy backing the Labels layer. Each frame slice is written
        # to the zarr store on save (avoids dask-write quirks while painting).
        self.masks_in_memory = np.zeros(
            (self.n_images, self.image_h, self.image_w), dtype=np.int16
        )

    def _create_layers(self):
        """Add the Image, Points and Labels layers to the viewer."""
        self.viewer.add_image(np.asarray(self.image_array), name="images")
        self.points_layer = self.viewer.add_points(
            np.empty((0, 3)),
            ndim=3,
            name="point prompts",
            face_color="red",
            border_color="white",
            size=20,
        )
        self.labels_layer = self.viewer.add_labels(
            self.masks_in_memory, name="masks"
        )

        # select the Points layer so the user can immediately click prompts
        self.viewer.layers.selection.active = self.points_layer
        self.points_layer.mode = "add"

    def _create_text_prompt_widget(self):
        """Text prompt line edit (empty -> geometric-only inference)."""
        self.text_prompt = QLineEdit()
        self.text_prompt.setText(DEFAULT_TEXT_PROMPT)
        self.layout().addRow("text prompt", self.text_prompt)

    def _create_inference_floor_widget(self):
        """SAM3 ``confidence_threshold`` used at inference time."""
        self.inference_floor = QDoubleSpinBox()
        self.inference_floor.setRange(0.0, 1.0)
        self.inference_floor.setSingleStep(0.05)
        self.inference_floor.setValue(DEFAULT_INFERENCE_FLOOR)
        self.inference_floor.valueChanged.connect(self._on_floor_changed)
        self.layout().addRow("inference floor", self.inference_floor)

    def _create_max_mask_frac_widget(self):
        """Drop masks larger than this fraction of the image."""
        self.max_mask_frac = QDoubleSpinBox()
        self.max_mask_frac.setRange(0.0, 1.0)
        self.max_mask_frac.setSingleStep(0.01)
        self.max_mask_frac.setValue(DEFAULT_MAX_MASK_FRAC)
        self.max_mask_frac.valueChanged.connect(self._on_postproc_changed)
        self.layout().addRow("max mask frac", self.max_mask_frac)

    def _create_min_mask_frac_widget(self):
        """Drop masks smaller than this fraction of the image."""
        self.min_mask_frac = QDoubleSpinBox()
        self.min_mask_frac.setDecimals(6)
        self.min_mask_frac.setRange(0.0, 1.0)
        self.min_mask_frac.setSingleStep(0.0005)
        self.min_mask_frac.setValue(DEFAULT_MIN_MASK_FRAC)
        self.min_mask_frac.valueChanged.connect(self._on_postproc_changed)
        self.layout().addRow("min mask frac", self.min_mask_frac)

    def _create_run_button(self):
        """Button that runs SAM3 on the frame in view."""
        self.run_button = QPushButton("Run inference")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.layout().addRow(self.run_button)

    def _create_threshold_widget(self):
        """Post-hoc confidence threshold slider (re-filters cached masks)."""
        self.threshold = QLabeledDoubleSlider()
        self.threshold.setRange(DEFAULT_INFERENCE_FLOOR, 1.0)
        self.threshold.setValue(DEFAULT_INFERENCE_FLOOR)
        self.threshold.valueChanged.connect(self._on_threshold_changed)
        self.layout().addRow("confidence threshold", self.threshold)

    def _connect_layer_events(self):
        """Auto-save manual edits on paint strokes and frame navigation."""
        self._prev_frame = self.current_frame()
        self.labels_layer.events.paint.connect(self._on_paint)
        self.viewer.dims.events.current_step.connect(self._on_step_change)

    # ------------------------------------------------------------------
    # Core operations

    def current_frame(self):
        """Frame index currently shown by the dims slider."""
        return int(self.viewer.dims.current_step[0])

    def points_for_frame(self, frame_idx):
        """``(M, 2)`` array of (x, y) pixel prompts placed on ``frame_idx``."""
        data = self.points_layer.data
        if len(data) == 0:
            return np.empty((0, 2), dtype=np.float32)
        on_frame = np.rint(data[:, 0]).astype(int) == frame_idx
        yx = data[on_frame, 1:]
        return yx[:, ::-1].astype(np.float32)  # (y, x) -> (x, y)

    def write_metadata(self):
        """Refresh the reproducibility metadata in the zarr store attrs."""
        self.mask_zarr.attrs.update(
            {
                "point_prompts_per_frame": {
                    str(k): np.asarray(v).tolist()
                    for k, v in self.point_prompts_per_frame.items()
                },
                "confidence_threshold_per_frame": {
                    str(k): float(v)
                    for k, v in self.threshold_per_frame.items()
                },
                "annotated_image_files": [
                    str(self.image_array.img_paths[i])
                    for i in sorted(self.annotated_frames)
                ],
                "text_prompt": self.text_prompt.text(),
                "inference_floor": float(self.inference_floor.value()),
                "max_mask_frac": float(self.max_mask_frac.value()),
                "min_mask_frac": float(self.min_mask_frac.value()),
            }
        )

    def save_frame(self, frame_idx):
        """Write one frame's mask slice to the zarr store, refresh metadata."""
        self.mask_zarr[frame_idx] = self.labels_layer.data[frame_idx]
        self.annotated_frames.add(frame_idx)
        self.write_metadata()

    def apply_threshold(self, frame_idx, thr=None):
        """Re-filter the cached SAM3 masks for a frame at confidence ``thr``.

        Pure post-hoc operation: no model call. Rebuilds the frame's mask from
        the cached SAM3 output, runs postprocessing, updates the Labels layer
        and saves. NOTE: this discards any manual paint edits on the frame --
        set the threshold first, then paint.
        """
        cache = self.inference_cache.get(frame_idx)
        if cache is None:
            return
        if thr is None:
            thr = self.threshold.value()

        masks_bool = cache["masks_bool"]
        scores = cache["scores"]
        keep = scores >= thr
        kept = masks_bool[keep]

        if len(kept) == 0:
            id_mask = np.zeros((self.image_h, self.image_w), dtype=np.int16)
        else:
            obj_ids = np.arange(1, len(kept) + 1, dtype=np.int16)[
                :, None, None
            ]
            id_mask = (kept.astype(np.int16) * obj_ids).max(axis=0)

        id_mask = postprocess(
            id_mask,
            frame_idx,
            self.max_mask_frac.value(),
            self.min_mask_frac.value(),
        )

        self.labels_layer.data[frame_idx] = id_mask.astype(np.int16)
        self.labels_layer.refresh()
        self.threshold_per_frame[frame_idx] = float(thr)
        self.save_frame(frame_idx)
        logger.info(
            "frame %d: %d masks at threshold %.2f",
            frame_idx,
            int(id_mask.max()),
            thr,
        )

    def run_inference(self):
        """Run SAM3 on the frame in view using its point prompts."""
        frame_idx = self.current_frame()
        points_xy = self.points_for_frame(frame_idx)
        if len(points_xy) == 0:
            logger.warning(
                "frame %d: no point prompts placed, skipping", frame_idx
            )
            return

        image = Image.fromarray(self.image_array[frame_idx])
        width, height = image.size

        # rebuild the processor so the inference floor from the UI is used
        processor = Sam3Processor(
            self.model,
            confidence_threshold=float(self.inference_floor.value()),
        )
        state = processor.set_image(image)
        processor.reset_all_prompts(state)

        text_prompt = self.text_prompt.text().strip()
        if text_prompt:
            state = processor.set_text_prompt(state=state, prompt=text_prompt)

        # normalize (x, y) pixel prompts to [0, 1] and add them one by one
        norm_xy = points_xy / np.array([width, height], dtype=np.float32)
        for px, py in norm_xy:
            state = add_point_prompt(
                processor, state, (float(px), float(py)), label=True
            )

        # move masks/scores to CPU, then release this frame's GPU state
        masks = state["masks"].cpu().numpy()
        scores = state["scores"].cpu().numpy().reshape(-1)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
        self.inference_cache[frame_idx] = {
            "masks_bool": masks.astype(bool),
            "scores": scores,
        }
        self.point_prompts_per_frame[frame_idx] = points_xy.tolist()
        logger.info(
            "frame %d: SAM3 returned %d masks (floor %.2f, %d point prompts)",
            frame_idx,
            masks.shape[0],
            self.inference_floor.value(),
            len(points_xy),
        )

        self.apply_threshold(frame_idx)

    def load_prompts_from_zarr(self, path):
        """Repopulate the Points layer + threshold dict from a saved store.

        Run this against an existing store to reproduce an analysis from its
        metadata. Returns the ``confidence_threshold_per_frame`` dict read from
        the store (also merged into ``self.threshold_per_frame``).
        """
        store = zarr.open(path, mode="r")
        prompts = store.attrs.get("point_prompts_per_frame", {})
        thresholds = store.attrs.get("confidence_threshold_per_frame", {})

        nd_points = []
        for frame_str, xy_list in prompts.items():
            frame = int(frame_str)
            for x, y in xy_list:  # stored as (x, y) pixels
                nd_points.append([frame, y, x])  # napari wants (frame, y, x)

        self.points_layer.data = (
            np.asarray(nd_points, dtype=np.float32)
            if nd_points
            else np.empty((0, 3), dtype=np.float32)
        )
        loaded = {int(k): float(v) for k, v in thresholds.items()}
        self.threshold_per_frame.update(loaded)
        logger.info("Loaded prompts for %d frames from %s", len(prompts), path)
        return loaded

    # ------------------------------------------------------------------
    # Callbacks

    def _on_run_clicked(self):
        self.run_inference()

    def _on_threshold_changed(self, value):
        self.apply_threshold(self.current_frame(), thr=value)

    def _on_floor_changed(self, value):
        # the post-hoc slider can never go below the inference floor
        self.threshold.setMinimum(value)
        if self.threshold.value() < value:
            self.threshold.setValue(value)

    def _on_postproc_changed(self, value):
        # re-run postprocessing on the current frame with the new size limits
        self.apply_threshold(self.current_frame())

    def _on_paint(self, event):
        self.save_frame(self.current_frame())

    def _on_step_change(self, event):
        new_frame = self.current_frame()
        if new_frame != self._prev_frame:
            self.save_frame(self._prev_frame)
            self._prev_frame = new_frame


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames as a lazy array

list_image_files = sorted(Path(images_dir).glob("*.png"))
image_array = ImageArrayLazy(list_image_files)
n_images, image_h, image_w = image_array.shape[:3]
print(f"Loaded {n_images} images of shape {(image_h, image_w)}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build SAM3 image model

# avoid bfloat16 / float mismatch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model()
print(f"SAM3 commit: {get_sam3_commit()}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create the viewer and attach the widget

viewer = napari.Viewer()
widget = Sam3BurrowWidget(viewer, image_array, model, OUTPUT_DIR)
viewer.window.add_dock_widget(widget, name="SAM3", area="right")

# Run the napari event loop (only needed when running as a plain script).
napari.run()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproducibility: reload point prompts from a saved zarr store
# Run this cell against an existing store to repopulate the Points layer and
# the per-frame thresholds, so an analysis can be reproduced from metadata.

# Example (uncomment and point at an existing store):
# widget.load_prompts_from_zarr(widget.output_masks_zarr)
