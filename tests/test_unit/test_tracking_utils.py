import csv
import io

import numpy as np
import pytest
import torch

from crabs.tracker.utils.tracking import (
    extract_bounding_box_info,
    write_tracked_bbox_to_csv,
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


@pytest.fixture
def csv_output():
    return io.StringIO()


@pytest.fixture
def csv_writer(csv_output):
    return csv.writer(csv_output)


def test_write_tracked_bbox_to_csv(csv_writer, csv_output):
    bbox = np.array([10, 20, 50, 80, 1])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_name = "frame_0001.png"
    pred_score = 0.900

    write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer, pred_score)

    expected_row = (
        "frame_0001.png",
        30000,
        '"{""clip"":123}"',
        1,
        0,
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":40,""height"":60}"',
        '"{""track"":""1"", ""confidence"":""0.9""}"',
    )
    expected_row_str = ",".join(map(str, expected_row))
    assert csv_output.getvalue().strip() == expected_row_str


@pytest.fixture
def mock_tracker():
    # Assuming mock_tracker has prep_sort method
    class MockTracker:
        def prep_sort(self, prediction):
            pred_boxes = prediction[0]["boxes"].detach().cpu().numpy()
            pred_scores = prediction[0]["scores"].detach().cpu().numpy()
            pred_labels = prediction[0]["labels"].detach().cpu().numpy()

            pred_sort = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if (
                    score > self.score_threshold
                ):  # Assuming score_threshold is a property of mock_tracker
                    bbox = np.concatenate((box, [score]))
                    pred_sort.append(bbox)

            return np.asarray(pred_sort)

    return MockTracker()


@pytest.fixture
def prediction():
    return [
        {
            "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            "scores": torch.tensor([0.8, 0.05]),
            "labels": torch.tensor([1, 2]),
        }
    ]


@pytest.fixture
def expected_output():
    return np.array([[10.0, 20.0, 30.0, 40.0, 0.8]])


@pytest.mark.parametrize(
    "score_threshold, prediction, expected_output",
    [
        (
            0.5,
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.8, 0.05]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array([[10.0, 20.0, 30.0, 40.0, 0.8]]),
        ),
        (
            0.3,
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.8, 0.4]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array(
                [[10.0, 20.0, 30.0, 40.0, 0.8], [50.0, 60.0, 70.0, 80.0, 0.4]]
            ),
        ),
        (
            0.6,
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.05, 0.05]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array([]),
        ),
    ],
)
def test_prep_sort(mock_tracker, score_threshold, prediction, expected_output):
    mock_tracker.score_threshold = score_threshold
    output = mock_tracker.prep_sort(prediction)

    assert np.allclose(
        output, expected_output
    ), "The output of prep_sort is not as expected."
    assert (
        output.shape == expected_output.shape
    ), "The shape of the output is incorrect."
    assert isinstance(output, np.ndarray), "The output type is incorrect."
