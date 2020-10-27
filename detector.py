from typing import List, Tuple

import numpy as np


class Detector:
    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detection = [
            (1, 1, 200, 200),
            (15, 15, 145, 155),
        ]
        return detection
