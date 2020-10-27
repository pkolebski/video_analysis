from typing import List, Tuple

import numpy as np
import tensorflow_hub as hub
import cv2


class Detector:
    def __init__(self, image_shape: Tuple[int, int] = (320, 320),
                 model_url: str = None) -> None:
        if model_url is None:
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

        self.model = hub.load(model_url)
        self.image_shape = image_shape

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.image_shape)
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        img = self.preprocess(img)
        output = self.model(img)
        detections = output['detection_boxes'][0].numpy()
        detections = [tuple() for x in detections]

        return detections
