# Review: branch `smg/gt-ellipses-prediction`

Scope: `git diff main...HEAD` (commits 90acde0..c5d779b), covering:

- `scripts/generate_masks_from_bboxes.py`
- `notebooks/notebook_visualise_masks_from_zarr.py`
- `tests/test_unit/test_generate_masks_from_bboxes.py`
- `claude-plans/proposal-mask-ellipse-params.md` (not reviewed as code)

Checks run before writing this:

- The 3 new unit tests pass (`pytest tests/test_unit/test_generate_masks_from_bboxes.py`).
- The theta conversion `np.mod(pi/2 - orientation, pi)` is correct. A mask along the down-right diagonal gives theta of about pi/4, so the angle runs from +x towards +y.
- The napari corner math and the matplotlib `Ellipse(angle=np.degrees(theta))` usage in the notebook both match that convention.
- No CLAUDE.md applies to this repo.

No crash-level bugs were found. The findings below are robustness, coverage and cleanup items, most important first.

## Instructions for the executor

Work through the findings in order. For each one, make the smallest change that resolves it. Keep the store layout described in the script docstring unless a finding says to change it. After the changes:

- run `pytest tests/test_unit/test_generate_masks_from_bboxes.py --no-cov`;
- run `ruff check scripts notebooks tests` and `ruff format` on the same paths.

Do not commit unless asked.

---

## 1. Image-id map keys change type after the round trip, and the notebook copies the lookup logic

- **Where:** `scripts/generate_masks_from_bboxes.py`: `_attrs_for_zarr` (~line 641) and `get_image_paths` (~line 148). `notebooks/notebook_visualise_masks_from_zarr.py` (~lines 226-232).
- **Problem:** `_attrs_for_zarr` leaves the int keys of `map_image_id_to_filename` as they are. zarr stores them as strings, so a reopened store has `{"5": "a.png"}`. Calling `get_image_paths` on a dataset opened from the store fails with a `KeyError`, because `image_id` values are ints and the keys are now strings. To work around this, the notebook copies the path-building logic with `int(k)` (the reuse problem in finding 6).
- **Fix:** handle this in one place. Either:
  - make `get_image_paths` accept both key types (look up `map_id_to_filename.get(img_id, map_id_to_filename.get(str(img_id)))`, or turn the keys back to `int` first); or
  - store the image filenames per frame as a data variable on `image_id`. `image_filenames` is already written as an attr.

  Then have the notebook call `get_image_paths(ds)` (import it from `scripts.generate_masks_from_bboxes`, as the tests do) instead of building the paths itself. Add a test that reopens the store and calls `get_image_paths` on it.

## 2. No tests for the overlap policy or the box-to-label mapping

- **Where:** `scripts/generate_masks_from_bboxes.py`: `flatten_masks_smallest_wins` (~line 663) and `get_bboxes_mask_labels_per_frame` (~line 550). `tests/test_unit/test_generate_masks_from_bboxes.py`.
- **Problem:** these two functions decide which pixel belongs to which box. If either is wrong, the masks and ellipse parameters land at the wrong `(image_id, id)` and nothing fails. For example, a change to the label offset or to the sort order in `argsort(-areas)` would silently assign shared pixels to the bigger crab. Neither function has a test. The proposal promised an eccentricity consistency check (`sqrt(1 - (minor/major)^2)`), and that test is also missing.
- **Fix:** add unit tests:
  - `flatten_masks_smallest_wins`: two overlapping masks of different sizes. The overlap takes the smaller mask's label. With equal areas, the order is stable.
  - `get_bboxes_mask_labels_per_frame`: a frame whose `id` slots are partly NaN (for example only slots 0 and 2 filled). It returns boxes in `[x1, y1, x2, y2]` order and labels `[1, 3]`. A frame with no boxes returns empty arrays.
  - eccentricity: check it against the axis lengths in `test_compute_mask_props`.

## 3. Box corners depend on the dimension order of `position`

- **Where:** `scripts/generate_masks_from_bboxes.py`, `get_bboxes_mask_labels_per_frame` (~lines 569-572).
- **Problem:** `x1y1.isel(image_id=idx).values.T` assumes that `position` and `shape` have dims `(image_id, space, id)`. If ethology reorders them, or `x1y1` broadcasts to a different order, `.T` returns `(space, id)` rows. The (N, 4) boxes are then scrambled, or the shape check fails only when SAM2 runs.
- **Fix:** use `x1y1.isel(image_id=idx).transpose("id", "space").values`, and the same for `x2y2`.

## 4. uint16 labels can overflow silently

- **Where:** `scripts/generate_masks_from_bboxes.py`: `get_bboxes_mask_labels_per_frame` (`astype(np.uint16) + 1`, ~line 579) and `create_dataset_store` (`label_of`, ~line 623).
- **Problem:** with more than 65535 `id` slots, the labels wrap to 0 (background), or collide with other labels, and nothing reports it. This is unlikely for crabs, but nothing guards against it.
- **Fix:** in `create_dataset_store`, assert `ds_bboxes.sizes["id"] < np.iinfo(np.uint16).max` with a clear error message.

## 5. Images with a different shape fail only after SAM2 has run

- **Where:** `scripts/generate_masks_from_bboxes.py`: `ImageArrayLazy.__init__` (~line 128) and `predict_masks_across_images` (`label_mask_batch[idx] = ...`).
- **Problem:** the store and the label batch are sized from the first image. An annotated image with a different size makes `flatten_masks_smallest_wins` return a mismatched array, and the assignment fails part way through the run, after SAM2 compute has been spent. The store is then left half-written. `ds_bboxes` already has a per-image `image_shape` variable (dims `(image_id, space)`) that could be checked up front. Separately, `__init__` reads `img_paths[0]` instead of `self.img_paths[0]`, which breaks if a generator is passed.
- **Fix:**
  - In `main`, before creating the store, check that all `ds_bboxes.image_shape` values are equal and match `(image_w, image_h)` from the first image. Raise an error otherwise.
  - In `ImageArrayLazy.__init__`, use `self.img_paths[0]`.

## 6. The notebook copies the lazy image loading that the script already has

- **Where:** `notebooks/notebook_visualise_masks_from_zarr.py`: `_read_image` and the `da.stack([da.from_delayed(...)])` block (~lines 82-85 and 222-246).
- **Problem:** the notebook rebuilds the image paths (the same logic as `get_image_paths`) and a lazy image array (the same job as `ImageArrayLazy`). It is a second copy of the image_id-to-file alignment, which has to stay in sync with the script.
- **Fix:** import `get_image_paths` from `scripts.generate_masks_from_bboxes` after finding 1 is fixed. Keep the dask stack if napari or xarray needs a real dask array, but build it from `get_image_paths(ds)`.

## 7. The mask parameters are computed by reading every frame back from disk

- **Where:** `scripts/generate_masks_from_bboxes.py`: `compute_mask_props` (~line 713) and `main` (~lines 888-890).
- **Problem:** `predict_masks_across_images` already has each label frame in memory (`label_mask_batch`). `compute_mask_props` then reads and decompresses every full-resolution frame from the zarr store again. On large datasets, that second pass over all frames is wasted I/O.
- **Fix (optional, a design choice):** compute the `regionprops_table` rows per frame inside the prediction loop, from `label_mask_batch`. Collect them into the same `(n_images, n_ids)` arrays and write `ds_props` at the end. To keep `compute_mask_props(labels, ds_bboxes)` testable, give it a per-frame helper that both paths call. If measuring on the stored masks is a deliberate guarantee, keep the current approach and say so in the docstring.

## 8. Until the last step, the store's consolidated metadata is stale

- **Where:** `scripts/generate_masks_from_bboxes.py`: `predict_masks_across_images` (`output_zarr.attrs["annotated_frames"] = ...`) and `main`.
- **Problem:** `create_dataset_store` writes consolidated metadata. The prediction loop then changes `labels` attrs through zarr directly, and `xr.open_zarr` does not see that change until the final `ds_props.to_zarr(mode="a")` rewrites the consolidated metadata. This was checked: before the props step, `labels.attrs` reads back as `{}`. If the run fails between the two steps, the store opens without error but has no `annotated_frames` and no `mask_*`/`ellipse_*` variables. The notebook then fails with an unclear `AttributeError`.
- **Fix:** either:
  - drop the `annotated_frames` attr, which is redundant now that every frame in `image_id` is processed; or
  - call `zarr.consolidate_metadata(output_zarr)` after prediction.

  Also consider writing an attr such as `"complete": True` only after `ds_props` is written, and checking it in the notebook.

## 9. Metadata attrs repeat what the dataset already stores

- **Where:** `scripts/generate_masks_from_bboxes.py`, `main`, `metadata_dict` (~lines 843-854).
- **Problem:**
  - `image_ids` repeats the `image_id` coordinate.
  - `image_filenames` repeats `map_image_id_to_filename`.
  - `n_images` and `image_shape` can be read from `labels.shape`. `image_shape` also has the same name as ethology's `image_shape` data variable, with a different meaning (one global `[h, w]` versus a per-image `(x, y)`).
  - `id_offset: 1` repeats `label_of`.

  Each duplicate can drift from the data it copies.
- **Fix:** remove `image_ids`, `n_images` and `image_shape` (or rename the last to `labels_shape`). Keep `image_filenames` only if finding 1 uses it as the source of the file paths.

## 10. The `space` cast to object lives in `main`, outside the store writer

- **Where:** `scripts/generate_masks_from_bboxes.py`: `main` (~line 822) and `create_dataset_store`. `tests/test_unit/test_generate_masks_from_bboxes.py` fixture (`# as main casts it`).
- **Problem:** the "zarr 3 has no stable format for fixed-width strings" workaround happens in `main`, so `create_dataset_store` depends on the caller to do it. The test fixture has to copy the cast by hand. A quick check shows the store still writes and reads back with `<U1` (it only warns), so the cast is a workaround for a warning and does not change behaviour. It belongs in the function that writes to zarr.
- **Fix:** move the `assign_coords(space=...astype(object))` into `create_dataset_store`, next to `_attrs_for_zarr`, and remove the cast from `main` and from the test fixture.

---

## Minor notes (no action required)

- `ellipse_major_axis_theta` means nothing for near-circular masks (`ellipse_eccentricity` about 0). scikit-image reports an arbitrary orientation there; a 2x2 square gives theta = 3*pi/4. Consider documenting this in the docstring, or setting theta to NaN below an eccentricity threshold.
- The layout in the script docstring says `label_of ... label = id + 1`. The label is the box's position along `id` plus 1, which equals `id + 1` only when the `id` coordinate is `0..n-1`. The function docstrings already say "position", so align the layout line with them.
- The docstring of `get_bboxes_mask_labels_per_frame` names the tuple `(boxes, mask labels)` but then refers to "``labels``"; use the same name in both places.
