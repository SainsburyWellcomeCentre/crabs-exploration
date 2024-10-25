"""SORT: A Simple, Online and Realtime Tracker.

Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Optional

import numpy as np


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Perform linear assignment.

    Uses LAPJV algorithm if available, otherwise falls back to scipy's
    linear_sum_assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray
        The cost matrix representing the assignment costs between
        tracks and detections.

    Returns
    -------
    np.ndarray
        An array containing the assignment indices. Each row corresponds to a
        pair (track index, detection index).

    """
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute IOU between two bboxes in the form [x1,y1,x2,y2].

    Calculate Intersection over Union (IoU) between two batches of
    bounding boxes.

    Parameters
    ----------
    bb_test : np.ndarray
        Bounding boxes of shape (N, 4) representing N test boxes
        in format [x1, y1, x2, y2].
    bb_gt : np.ndarray
        Bounding boxes of shape (M, 4) representing M ground truth
        boxes in format [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        IoU values between each pair of bounding boxes in bb_test and bb_gt.
        The shape of the returned array is (N, M).

    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0])
        * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert a bounding box from corner form to center form.

    Corner form is [x1, y1, x2, y2] and center form is [x, y, s, r].

    Parameters
    ----------
    bbox : np.ndarray
        Bounding box coordinates in the form [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        Converted representation of the bounding box as [x, y, s, r].
        T

    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(
    x: np.ndarray, score: Optional[float] = None
) -> np.ndarray:
    """Convert a bounding box from center form to corner form.

    Center form is [x, y, s, r] and corner form is [x1, y1, x2, y2].

    Parameters
    ----------
    x : np.ndarray
        Bounding box coordinates in the center form [x, y, s, r].
    score : float, optional
        Optional score associated with the bounding box.

    Returns
    -------
    np.ndarray
        Converted representation of the bounding box as [x1, y1, x2, y2]
        (and score, if provided).
        The shape of the returned array is (1, 4) or (1, 5)
        if score is provided.

    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0,
                score,
            ]
        ).reshape((1, 5))


def associate_detections_to_trackers(  # noqa: C901
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign detections to tracked objects.

    Both detections and tracked objects are represented as bounding boxes.

    Parameters
    ----------
    detections : np.ndarray
        Array of shape (N, 4) representing N detection bounding boxes in
        format [x1, y1, x2, y2].
    trackers : np.ndarray
        Array of shape (M, 4) representing M tracker bounding boxes in
        format [x1, y1, x2, y2].
    iou_threshold : float, optional
        IOU threshold for associating detections with trackers. Default is 0.3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Three arrays:
        - matches: Array of shape (K, 2) containing indices of matched
        detections and trackers.
        - unmatched_detections: Array of indices of detections that were not
        matched.
        - unmatched_trackers: Array of indices of trackers that were not
        matched.

    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = []
    for d, _det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    list_matches = []  # before: matches
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            list_matches.append(m.reshape(1, 2))
    if len(list_matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(list_matches, axis=0)

    return (
        matches,
        np.array(unmatched_detections),
        np.array(unmatched_trackers),
    )
