import abc
from typing import List, Tuple

import numpy as np
import cv2

from detectors.detector import Detection


class BaseTracker(abc.ABC):
    def __init__(self):
        self.active_tracked: List[List[int, bool, List[Detection]]] = []

    @abc.abstractmethod
    def match_bbs(self, bbs: List[Detection], frame=None) -> None:
        pass

    def plot_history(self, frame: np.ndarray):
        if type(self.active_tracked) is dict:
            for key, val in self.active_tracked.items():
                np.random.seed(key)
                color = (list(np.random.choice(range(256), size=3)))
                color = [int(color[0]), int(color[1]), int(color[2])]
                last_history = val[2][:]
                for i, det in enumerate(last_history):
                    if i > 0:
                        cv2.line(
                            frame,
                            last_history[i - 1].get_center(),
                            last_history[i].get_center(),
                            color,
                            thickness=2
                        )
                frame = cv2.putText(frame,
                                    str(key),
                                    val[2][len(val[2]) - 1].get_center(),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2)
            return frame

        for _, _, history in self.active_tracked:
            last_history = history[:]
            for i, det in enumerate(last_history):
                if i > 0:
                    cv2.line(
                        frame,
                        last_history[i - 1].get_center(),
                        last_history[i].get_center(),
                        (0, 355, 0),
                        thickness=2
                    )
        return frame
