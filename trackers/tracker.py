import abc
from typing import List, Tuple

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

from detectors.detector import Detection


class BaseTracker(abc.ABC):
    def __init__(self):
        self.active_tracked: List[List[int, bool, List[Detection]]] = []

    @abc.abstractmethod
    def match_bbs(self, bbs: List[Detection], frame=None) -> None:
        pass

    def plot_history(self, frame: np.ndarray, heatmap: bool = False):
        if heatmap:
            for key, val in (self.inactive_tracked.items() if type(self.inactive_tracked) == dict else enumerate(self.inactive_tracked)):
                np.random.seed(key)
                color = (list(np.random.choice(range(256), size=3)))
                color = [int(color[0]), int(color[1]), int(color[2])]
                last_history = [d.get_center() for d in val[2][:]]

                x, y = zip(*last_history)
                t = np.linspace(0, 1, len(x))
                t2 = np.linspace(0, 1, 100)

                x2 = np.interp(t2, t, x)
                y2 = np.interp(t2, t, y)
                sigma = 10
                x3 = gaussian_filter1d(x2, sigma)
                y3 = gaussian_filter1d(y2, sigma)

                x4 = np.interp(t, t2, x3)
                y4 = np.interp(t, t2, y3)

                for i, (l_x, l_y) in enumerate(zip(x4, y4)):
                    if i > 0:
                        cv2.line(
                            frame,
                            (int(x4[i - 1]), int(y4[i - 1])),
                            (int(l_x), int(l_y)),
                            color,
                            thickness=2
                        )
            return
        for key, val in (self.active_tracked.items() if type(self.active_tracked) is dict else enumerate(self.active_tracked)):
            np.random.seed(key)
            color = (list(np.random.choice(range(256), size=3)))
            color = [int(color[0]), int(color[1]), int(color[2])]
            last_history = [d.get_center() for d in val[2][:]]

            x, y = zip(*last_history)
            t = np.linspace(0, 1, len(x))
            t2 = np.linspace(0, 1, 100)

            x2 = np.interp(t2, t, x)
            y2 = np.interp(t2, t, y)
            sigma = 10
            x3 = gaussian_filter1d(x2, sigma)
            y3 = gaussian_filter1d(y2, sigma)

            x4 = np.interp(t, t2, x3)
            y4 = np.interp(t, t2, y3)

            for i, (l_x, l_y) in enumerate(zip(x4, y4)):
                if i > 0:
                    cv2.line(
                        frame,
                        (int(x4[i - 1]), int(y4[i - 1])),
                        (int(l_x), int(l_y)),
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
