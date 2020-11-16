import abc
from typing import List

import numpy as np
import cv2

from detectors.detector import Detection


class BaseTracker(abc.ABC):
    def __init__(self):
        self.active_tracked: List[List[Detection]] = []

    @abc.abstractmethod
    def match_bbs(self, bbs: List[Detection]) -> None:
        pass

    def plot_history(self, frame: np.ndarray):
        for history in self.active_tracked:
            last_history = history[:]
            for i, det in enumerate(last_history):
                if i > 0:
                    cv2.line(
                        frame,
                        last_history[i - 1].get_center(),
                        det.get_center(),
                        (0, 355, 0),
                        thickness=2
                    )
        return frame