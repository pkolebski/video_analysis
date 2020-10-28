
from typing import List, Tuple
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from utils.download import download_url
from detectors.detector import BaseDetector
import cv2

OBJECTS_MAP = {2: "Bicycle", 3: "Car", 4: "Motorcycle", 6: "Bus", 7: "Truck"}
PATH_MODEL = "pretrained_models"

class Yolov3(BaseDetector):
    def __init__(self, threshold: float = 0.3,
                 model_weights: str = "https://pjreddie.com/media/files/yolov3-tiny.weights",
                 model_cfg: str = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"):
        super().__init__(object_map=OBJECTS_MAP)
        self.threshold = threshold
        download_url(model_cfg, PATH_MODEL+"/yolov3-tiny.cfg")
        download_url(model_weights, PATH_MODEL+"/yolov3-tiny.weights")
        self.model = cv2.dnn.readNet(PATH_MODEL+"/yolov3-tiny.weights", PATH_MODEL+"/yolov3-tiny.cfg")
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.image_shape)
        img = np.expand_dims(img, axis=0)
        return img

    def detect_with_desc(self, img: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        img_shape = img.shape[:2]
        #0.00392 = 1/255

        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.output_layers)
        detections = []

        for output in outs:
            for detect in output:

                #   if detection_class in self.object_map.keys():
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > self.threshold and class_id in self.object_map.keys():
                    center_x = int(detect[0] * img_shape[1])
                    center_y = int(detect[1] * img_shape[0])
                    w = int(detect[2]*img_shape[1])
                    h = int(detect[3] * img_shape[0])
                    positions = (int(center_x-w/2), int(center_y-h/2), int(center_x+w/2), int(center_y+h/2))
                    box = positions + (self.object_map[class_id], conf)
                    detections.append(box)
        return detections
