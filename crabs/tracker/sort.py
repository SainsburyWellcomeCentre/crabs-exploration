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

import numpy as np
from filterpy.kalman import KalmanFilter

from crabs.tracker.utils.sort import (
    associate_detections_to_trackers,
    convert_bbox_to_z,
    convert_x_to_bbox,
)


class KalmanBoxTracker:
    """Class for the internal state of individual tracked objects.

    Parameters
    ----------
    bbox : np.ndarray
        Initial bounding box coordinates in the format [x1, y1, x2, y2].

    """

    count = 0

    def __init__(self, bbox):
        """Initialise a tracker using initial bounding box."""
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
        self.kf.P[4:, 4:] *= (
            1000.0
            # give high uncertainty to the unobservable initial velocities
        )
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
        """Update the state vector with an observed bounding box.

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
        """Advance the state vector and return predicted bounding box estimate.

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
        """Return the current bounding box estimate.

        Returns
        -------
        np.ndarray
            Current bounding box coordinates in the format [x1, y1, x2, y2].

        """
        return convert_x_to_bbox(self.kf.x)


class Sort:  # noqa: D101
    def __init__(
        self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3
    ):
        """Set key parameters for SORT.

        Parameters
        ----------
        max_age : int, optional
            Maximum number of frames to keep a tracker alive without an update.
            Default is 1.
        min_hits : int, optional
            Minimum number of consecutive hits to consider a tracker valid.
            Default is 3.
        iou_threshold : float, optional
            IOU threshold for associating detections with trackers.
            Default is 0.3.

        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list = []
        self.frame_count = 0

    def update(
        self,
        dets: np.ndarray = np.empty((0, 5)),  # noqa: B008
    ) -> np.ndarray:
        """Update the SORT tracker with new detections.

        Parameters
        ----------
        dets : np.ndarray, optional
            Array of shape (N, 5) representing N detection bounding boxes in
            format [x1, y1, x2, y2, score]. Use np.empty((0, 5)) for frames
            without detections.

        Returns
        -------
        np.ndarray
            Array of tracked objects with object IDs added as the last column.
            The shape of the array is (M, 5), where M is the number of tracked
            objects.

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
