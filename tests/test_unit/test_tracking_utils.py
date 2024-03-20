from crabs.detection_tracking.tracking_utils import count_identity_switches


def test_count_identity_switches():
    prev_frame_id = None
    current_frame_id = [[6, 5, 4, 3, 2, 1]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 0

    # Test with no identity switches
    prev_frame_id = [[6, 5, 4, 3, 2, 1]]
    current_frame_id = [[6, 5, 4, 3, 2, 1]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 0

    prev_frame_id = [[5, 6, 4, 3, 1, 2]]
    current_frame_id = [[6, 5, 4, 3, 2, 1]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 0

    prev_frame_id = [[6, 5, 4, 3, 2, 1]]
    current_frame_id = [[6, 5, 4, 2, 1]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 1

    prev_frame_id = [[6, 5, 4, 2, 1]]
    current_frame_id = [[6, 5, 4, 2, 1, 7]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 1

    prev_frame_id = [[6, 5, 4, 2, 1, 7]]
    current_frame_id = [[6, 5, 4, 2, 7, 8]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 2

    prev_frame_id = [[6, 5, 4, 2, 7, 8]]
    current_frame_id = [[6, 5, 4, 2, 7, 8, 3]]
    assert count_identity_switches(prev_frame_id, current_frame_id) == 1
