# sort.py
# Simple Online and Realtime Tracking (SORT)
# Works with YOLOv8 / YOLOv11 and your head detection function.

from __future__ import print_function
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# -------------------------------------------------------------
#    SORT: A Simple Kalman Filter + Hungarian Algorithm Tracker
# -------------------------------------------------------------

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6)
    return o


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        """
        bbox: (x1, y1, x2, y2)
        """
        # Create a Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = 1
        self.kf.F[1, 5] = 1
        self.kf.F[2, 6] = 1
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:, :4] = np.eye(4)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        self.kf.x[:4] = bbox.reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        """
        Update KF with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape((4, 1)))

    def predict(self):
        """
        Run KF prediction.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Return current bounding box.
        """
        x = self.kf.x
        return np.array([x[0], x[1], x[2], x[3]]).reshape((4,))


class Sort(object):
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        """
        max_age: frames to keep tracker without detection
        min_hits: required frames before tracker is valid
        iou_threshold: minimum IOU for assignment
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: [[x1, y1, x2, y2, score], ...]
        Returns: tracked boxes with ID → [x1,y1,x2,y2,ID]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))

        to_del = []
        ret = []

        # Predict new tracker positions
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = pos
            trks[t][4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        dets = dets.astype(float)

        if dets.shape[0] > 0:
            # IOU matrix
            iou_matrix = np.zeros((trks.shape[0], dets.shape[0]), dtype=np.float32)

            for t, trk in enumerate(trks):
                for d, det in enumerate(dets):
                    iou_matrix[t, d] = iou(det, trk)

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            unmatched_trks = []
            for t in range(trks.shape[0]):
                if t not in row_ind:
                    unmatched_trks.append(t)

            unmatched_dets = []
            for d in range(dets.shape[0]):
                if d not in col_ind:
                    unmatched_dets.append(d)

            # Assign matches
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    self.trackers[r].update(dets[c, :4])

            # New trackers for unmatched detections
            for idx in unmatched_dets:
                trk = KalmanBoxTracker(dets[idx, :4])
                self.trackers.append(trk)

        # Output tracked results
        for trk in self.trackers:
            if trk.time_since_update < 1 and \
               (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):

                d = trk.get_state()
                ret.append(np.concatenate((d, [trk.id])))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return np.array(ret)
