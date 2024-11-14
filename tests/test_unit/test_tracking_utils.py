import pytest
import torch

from crabs.tracker.utils.tracking import (
    extract_bounding_box_info,
    format_and_filter_bbox_predictions_for_sort,
)


def test_extract_bounding_box_info():
    csv_row = [
        "frame_00000001.png",
        "26542080",
        "{" "clip" ":123}",
        "1",
        "0",
        '{"name":"rect","x":2894.860594987354,"y":975.8516839863181,"width":51,"height":41}',
        '{"track":"79.0"}',
    ]

    result = extract_bounding_box_info(csv_row)

    expected_result = {
        "frame_number": 1,
        "x": 2894.860594987354,
        "y": 975.8516839863181,
        "width": 51,
        "height": 41,
        "id": "79.0",
    }

    assert result == expected_result


@pytest.mark.parametrize(
    "score_threshold, expected_output",
    [
        (
            0.5,
            torch.tensor(
                [
                    [10, 20, 30, 40, 0.9],
                    [50, 60, 70, 80, 0.85],
                    [15, 25, 35, 45, 0.8],
                ]
            ),
        ),
        (
            0.83,
            torch.tensor(
                [
                    [10, 20, 30, 40, 0.9],
                    [50, 60, 70, 80, 0.85],
                ]
            ),
        ),
        (
            0.95,
            torch.empty((0, 5)),
        ),
    ],
)
def test_format_bbox_predictions_for_sort(score_threshold, expected_output):
    # Define the test data
    prediction = {
        "boxes": torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80], [15, 25, 35, 45]]
        ),
        "scores": torch.tensor([0.9, 0.85, 0.8]),
    }

    # Call the function
    result = format_and_filter_bbox_predictions_for_sort(
        prediction, score_threshold
    )

    # Assert the result
    (
        torch.testing.assert_close(result, expected_output),
        f"Expected {expected_output}, but got {result}",
    )
