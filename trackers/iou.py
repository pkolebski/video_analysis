from typing import List

import numpy as np
import cv2

from trackers.tracker import BaseTracker
from detectors.detector import Detection


class IouTracker(BaseTracker):
    def __init__(self, min_track_threshold: float = .2, max_frames_track: float = 10):
        super(IouTracker, self).__init__()

        self.min_track_threshold: float = min_track_threshold
        self.max_frames_track: float = max_frames_track
        self.inactive_tracked = []

    def match_bbs(self, bbs: List[Detection], frame=None) -> None:
        for i in range(len(self.active_tracked)):
            self.active_tracked[i][1] = False
        active_tracked = self.active_tracked[:]
        a = 1
        for bb1 in bbs:
            ious = []
            for i, (frames_since_update, is_updated, bb2) in enumerate(active_tracked):
                bb2 = bb2[-1]
                ious.append(calculate_iou(bb1, bb2))
            if len(ious) > 0:
                ind = int(np.argmax(np.array(ious)))
                if ious[ind] > self.min_track_threshold:
                    history = self.active_tracked[ind][2]
                    bb1 = momentum_step(history[-1], bb1)
                    history.append(bb1)
                    self.active_tracked[ind] = [0, True, history]
                else:
                    self.active_tracked.append([0, True, [bb1]])
            else:
                self.active_tracked.append([0, True, [bb1]])

        for i in range(len(self.active_tracked)):
            if not self.active_tracked[i][1]:
                self.active_tracked[i][0] += 1
                if self.active_tracked[i][0] > self.max_frames_track:
                    self.active_tracked[i][2] = []

        self.inactive_tracked = [x for x in self.active_tracked if not x[2]]
        self.active_tracked = [x for x in self.active_tracked if x[2]]


def calculate_iou(bb1: Detection, bb2: Detection) -> float:
    xA = max(bb1.position[0], bb2.position[0])
    yA = max(bb1.position[1], bb2.position[1])
    xB = min(bb1.position[2], bb2.position[2])
    yB = min(bb1.position[3], bb2.position[3])

    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    bb1_area = (bb1.position[2] - bb1.position[0] + 1) * \
               (bb1.position[3] - bb1.position[1] + 1)
    bb2_area = (bb2.position[2] - bb2.position[0] + 1) * \
               (bb2.position[3] - bb2.position[1] + 1)

    iou = intersection / (bb1_area + bb2_area - intersection)
    return iou


def momentum_step(h1: Detection, h2: Detection, momentum: float = 0.9):
    a = [x1 - x2 for x1, x2 in zip(h2.get_center(), h1.get_center())]
    a = a[0], a[1], a[0], a[1]
    p = [x1 + x2 for x1, x2 in zip(h1.position, a)]
    new_point = [int(momentum * x1 + (1 - momentum) * x2) for x1, x2 in zip(p, h2.position)]
    h2.position = new_point
    return h2
