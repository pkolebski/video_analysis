from typing import List

import numpy as np
import cv2

from trackers.tracker import BaseTracker
from detectors.detector import Detection


class IouTracker(BaseTracker):
    def __init__(self, min_track_threshold: float = .2):
        super(IouTracker, self).__init__()

        self.min_track_threshold: float = min_track_threshold

    def match_bbs(self, bbs: List[Detection]) -> None:
        active_tracked = self.active_tracked[:]
        for bb1 in bbs:
            ious = []
            for i, bb2 in enumerate(active_tracked):
                bb2 = bb2[-1]
                ious.append(calculate_iou(bb1, bb2))
            if len(ious) > 0:
                ind = int(np.argmax(np.array(ious)))
                if ious[ind] > self.min_track_threshold:
                    self.active_tracked[ind].append(bb1)
                else:
                    self.active_tracked.append([bb1])
            else:
                self.active_tracked.append([bb1])


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
