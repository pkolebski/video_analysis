from typing import List, Tuple, Dict
import numpy as np
import abc
from dataclasses import dataclass


@dataclass
class Detection:
    position: Tuple[int, int, int, int]
    obj_type: str
    probability: float


class BaseDetector(abc.ABC):
    def __init__(self, object_map: Dict = None):
        self.object_map = object_map

    @abc.abstractmethod
    def detect_with_desc(self, img: np.ndarray) -> List[Detection]:
        """
        return boxes with
        :param img: image
        :rtype: [x, y, x, y, obj_type, probability]
        """
        pass

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detections = self.detect_with_desc(img)
        return [x.position[0:4] for x in detections]

    def index_to_vehicle_type(self, index: int):
        if self.object_map is None:
            raise RuntimeError("object_map isnt implemented")

    def fit(self, data_x, data_y):
        raise RuntimeError("fit method not implemented")


class ExampleDetector(BaseDetector):
    def __init__(self, object_map):
        super().__init__(object_map)

    def detect_with_desc(self, img: np.ndarray) -> List[Detection]:
        return [Detection((1, 1, 200, 200), 'car', 0.4),
                Detection((300, 300, 350, 600), 'bus', 0.4)]
