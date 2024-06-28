"""
SORT: A Simple, Online and Realtime Tracker
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

from typing import Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Perform linear assignment using LAPJV algorithm if available, otherwise fallback to scipy's linear_sum_assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray
        The cost matrix representing the assignment costs between tracks and detections.

    Returns
    -------
    np.ndarray
        An array containing the assignment indices. Each row corresponds to a pair (track index, detection index).
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
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    Calculate Intersection over Union (IoU) between two batches of bounding boxes.

    Parameters
    ----------
    bb_test : np.ndarray
        Bounding boxes of shape (N, 4) representing N test boxes in format [x1, y1, x2, y2].
    bb_gt : np.ndarray
        Bounding boxes of shape (M, 4) representing M ground truth boxes in format [x1, y1, x2, y2].

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
    """
    Convert a bounding box from [x1, y1, x2, y2] to a representation [x, y, s, r].

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
    """
    Convert a bounding box from center form [x, y, s, r] to corner form [x1, y1, x2, y2].

    Parameters
    ----------
    x : np.ndarray
        Bounding box coordinates in the center form [x, y, s, r].
    score : float, optional
        Optional score associated with the bounding box.

    Returns
    -------
    np.ndarray
        Converted representation of the bounding box as [x1, y1, x2, y2] (and score, if provided).
        The shape of the returned array is (1, 4) or (1, 5) if score is provided.
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


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects
    observed as bbox.

    Parameters
    ----------
    bbox : np.ndarray
        Initial bounding box coordinates in the format [x1, y1, x2, y2].

    """

    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray) -> None:
        """
        Updates the state vector with an observed bounding box.

        Parameters
        ----------
        bbox : np.ndarray
            Observed bounding box coordinates in the format [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Returns
        -------
        np.ndarray
            Predicted bounding box coordinates in the format [x1, y1, x2, y2].
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.

        Returns
        -------
        np.ndarray
            Current bounding box coordinates in the format [x1, y1, x2, y2].
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked objects (both represented as bounding boxes).

    Parameters
    ----------
    detections : np.ndarray
        Array of shape (N, 4) representing N detection bounding boxes in format [x1, y1, x2, y2].
    trackers : np.ndarray
        Array of shape (M, 4) representing M tracker bounding boxes in format [x1, y1, x2, y2].
    iou_threshold : float, optional
        IOU threshold for associating detections with trackers. Default is 0.3.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Three arrays:
        - matches: Array of shape (K, 2) containing indices of matched detections and trackers.
        - unmatched_detections: Array of indices of detections that were not matched.
        - unmatched_trackers: Array of indices of trackers that were not matched.
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
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return (
        matches,
        np.array(unmatched_detections),
        np.array(unmatched_trackers),
    )


class Sort(object):
    def __init__(
        self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3
    ):
        """
        Sets key parameters for SORT.

        Parameters
        ----------
        max_age : int, optional
            Maximum number of frames to keep a tracker alive without an update. Default is 1.
        min_hits : int, optional
            Minimum number of consecutive hits to consider a tracker valid. Default is 3.
        iou_threshold : float, optional
            IOU threshold for associating detections with trackers. Default is 0.3.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list = []
        self.frame_count = 0

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """
        Updates the SORT tracker with new detections.

        Parameters
        ----------
        dets : np.ndarray, optional
            Array of shape (N, 5) representing N detection bounding boxes in format [x1, y1, x2, y2, score].
            Use np.empty((0, 5)) for frames without detections.

        Returns
        -------
        np.ndarray
            Array of tracked objects with object IDs added as the last column.
            The shape of the array is (M, 5), where M is the number of tracked objects.
        """

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        (
            matched,
            unmatched_dets,
            unmatched_trks,
        ) = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits
                or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

