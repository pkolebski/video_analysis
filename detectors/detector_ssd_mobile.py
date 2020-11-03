from typing import List, Tuple
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from detectors.detector import Detection

from detectors.detector import BaseDetector
import cv2

OBJECTS_MAP = {2: "Bicycle", 3: "Car", 4: "Motorcycle", 6: "Bus", 7: "Truck"}


class DetectorSSDMobileNetv2(BaseDetector):
    def __init__(self, threshold: float = 0.3, model_url: str = None, image_shape = (320, 320)):
        super().__init__(object_map=OBJECTS_MAP)
        if model_url is None:
            self.model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
        self.model = hub.load(self.model_url)
        self.threshold = threshold
        self.image_shape = image_shape

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.image_shape)
        img = np.expand_dims(img, axis=0)
        return img

    def detect_with_desc(self, img: np.ndarray) -> List[Detection]:
        img_shape = img.shape[:2]
        img = self.preprocess(img)
        output = self.model(img)
        detections = []

        for i, detection_class in enumerate(output['detection_classes'].numpy()[0]):
            if detection_class in self.object_map.keys():
                if output['detection_scores'][0][i] > self.threshold:
                    box = output['detection_boxes'][0][i]
                    vehicle_type = self.object_map[detection_class]
                    prob = output['detection_scores'][0][i].numpy()
                    positions = np.array(box * (img_shape + img_shape))
                    # y,x,y,x to x,y,x,y
                    positions = (positions[1], positions[0], positions[3], positions[2])
                    detection = Detection(positions, vehicle_type, prob)
                    detections.append(detection)
                else:
                    return detections
        return detections
