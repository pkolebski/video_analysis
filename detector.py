from typing import List, Tuple

import numpy as np
import tensorflow_hub as hub
import cv2

HUMAN = 1
BICYCLE = 2
CAR = 3
MOTORCYCLE = 4
BUS = 6
TRUCK = 8

class Detector:
    def __init__(self, image_shape: Tuple[int, int] = (320, 320), threshold: float = 0.35,
                 model_url: str = None) -> None:
        if model_url is None:
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

        self.threshold = threshold
        self.model = hub.load(model_url)
        self.image_shape = image_shape
        self.objects = [BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.image_shape)
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        img_shape = img.shape[:2]
        img = self.preprocess(img)
        output = self.model(img)
        detections = []

        for i, detection_class in enumerate(output['detection_classes'].numpy()[0]):
            if detection_class in self.objects:
                if output['detection_scores'][0][i] > self.threshold:
                    box = output['detection_boxes'][0][i]
                    box = tuple(box*(img_shape+img_shape))
                    detections.append(box)
                else:
                    return detections
        return detections
