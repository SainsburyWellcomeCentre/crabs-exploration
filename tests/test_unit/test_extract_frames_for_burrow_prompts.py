import numpy as np
import pytest
import xarray as xr

from crabs.utils.compute_frames_for_burrow_prompts import (
    _counts_per_video_frame,
    _select_lowest_count_frame_idcs,
    _video_idcs_to_per_clip_idcs,
)


@pytest.fixture
def sample_video_dataset():
    """Build a synthetic per-video Dataset with two clips of 5 frames each.

    Confidence has dims (clip, frame, individual) with deliberate NaNs so
    the per-frame non-null counts are known in advance.
    """
    confidence = np.array(
        [
            # clip 0
            [
                [1.0, 1.0, 1.0],  # frame 0 -> 3 detections
                [1.0, 1.0, np.nan],  # frame 1 -> 2 detections
                [1.0, np.nan, np.nan],  # frame 2 -> 1 detections
                [np.nan, np.nan, np.nan],  # frame 3 -> 0 detections
                [1.0, 1.0, 1.0],  # frame 4 -> 3 detections
            ],
            # clip 1
            [
                [1.0, np.nan, np.nan],  # frame 0 -> 1 detections
                [1.0, 1.0, np.nan],  # frame 1 -> 2 detections
                [1.0, 1.0, np.nan],  # frame 2 -> 2 detections
                [1.0, 1.0, 1.0],  # frame 3 -> 3 detections
                [np.nan, np.nan, np.nan],  # frame 4 -> 0 detections
            ],
        ]
    )
    expected_counts = np.array([3, 2, 1, 0, 3, 1, 2, 2, 3, 0])

    ds = xr.Dataset(
        data_vars={
            "confidence": (
                ("clip_id", "frame", "individual"),
                confidence,
            ),
            "clip_first_frame_0idx": (("clip_id",), np.array([0, 5])),
            "clip_last_frame_0idx": (("clip_id",), np.array([4, 9])),
        },
        coords={
            "clip_id": np.array(["clip0", "clip1"]),
        },
    )
    return ds, expected_counts


def test_counts_per_video_frame(sample_video_dataset):
    """Check the computed counts per video frame matches dataset."""
    ds, expected_counts = sample_video_dataset
    counts = _counts_per_video_frame(ds)

    # check total number of frames
    assert len(counts) == ds.clip_last_frame_0idx.isel(clip_id=-1).item() + 1
    # check number of detections per frame
    np.testing.assert_array_equal(counts, expected_counts)


def test_video_idcs_to_per_clip_idcs(sample_video_dataset):
    """Check frame indices conversion from clip-based to video-based."""
    ds_video, _ = sample_video_dataset
    counts_video = _counts_per_video_frame(ds_video)

    # Compute clip based indices
    frame_based_idcs = _select_lowest_count_frame_idcs(
        counts_video, frames_fraction=0.5
    )  # sorted by count
    clip_based_idcs = _video_idcs_to_per_clip_idcs(frame_based_idcs, ds_video)

    # Compute frame-based indices from clip-based ones
    frame_based_idcs_from_clip = np.concatenate(
        [
            clip_based_idcs[clip_id]
            + ds_video.clip_first_frame_0idx.sel(clip_id=clip_id).values
            for clip_id in ds_video.clip_id.values
            if clip_id in clip_based_idcs
        ]
    )  # not sorted by count

    # To compare we ignore sorting
    # - frame_based_idcs is sorted by ascending count
    # - frame_based_idcs_from_clip is not sorted by count
    assert set(frame_based_idcs) == set(frame_based_idcs_from_clip)


def test_select_lowest_count_frame_idcs():
    """The selected indices correspond to the n smallest counts and are
    returned in ascending count order.
    """
    counts = np.array([5, 1, 3, 2, 4, 1, 6, 0, 7, 2])
    frames_fraction = 0.5

    idcs = _select_lowest_count_frame_idcs(counts, frames_fraction)

    # Check number of extracted frames
    n_to_extract = int(len(counts) * frames_fraction)
    assert idcs.shape == (n_to_extract,)

    # Check extracted frames are bottom ones after sorting
    np.testing.assert_array_equal(
        counts[idcs],
        np.sort(counts)[:n_to_extract],
    )
