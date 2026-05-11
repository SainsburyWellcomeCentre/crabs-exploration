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


def test_counts_per_video_frame_length_matches_clip_last_frame(
    sample_video_dataset,
):
    """Replaces the notebook assert at lines 119-123."""
    ds, expected_counts = sample_video_dataset

    counts = _counts_per_video_frame(ds)

    assert len(counts) == ds.clip_last_frame_0idx.isel(clip_id=-1).item() + 1
    np.testing.assert_array_equal(counts, expected_counts)


def test_video_idcs_round_trip_with_per_clip_idcs():
    """Replaces the notebook assert at lines 130-152.

    Selecting bottom-N video-frame indices and then splitting them into
    per-clip indices must round-trip back to the original video-frame
    indices when we add each clip's start offset.
    """
    counts_video = np.array([5, 1, 3, 2, 4, 6, 0, 7, 2, 1])
    n_frames_per_clip = np.array([5, 5])
    clip_ids = np.array(["c0", "c1"])
    clip_first_frame_0idx = np.array([0, 5])
    n_to_extract = 4

    video_idcs = _select_lowest_count_frame_idcs(counts_video, n_to_extract)
    per_clip = _video_idcs_to_per_clip_idcs(
        video_idcs, n_frames_per_clip, clip_ids
    )

    reconstructed = np.concatenate(
        [
            per_clip[clip_id] + clip_first_frame_0idx[i]
            for i, clip_id in enumerate(clip_ids)
            if clip_id in per_clip
        ]
    )

    np.testing.assert_array_equal(np.sort(video_idcs), np.sort(reconstructed))


def test_select_lowest_count_frame_idcs_picks_smallest():
    """The selected indices correspond to the n smallest counts and are
    returned in ascending count order.
    """
    counts = np.array([5, 1, 3, 2, 4, 1, 6, 0, 7, 2])
    n_to_extract = 4

    idcs = _select_lowest_count_frame_idcs(counts, n_to_extract)

    assert idcs.shape == (n_to_extract,)
    np.testing.assert_array_equal(
        np.sort(counts[idcs]), np.sort(counts)[:n_to_extract]
    )
    assert np.all(np.diff(counts[idcs]) >= 0)
