from typing import List, Tuple
import numpy as np
from utils.download import download_url
from detectors.detector import BaseDetector, Detection
import cv2

PATH_MODEL = "pretrained_models"


class Yolo(BaseDetector):
    def __init__(self,
                 objects_map: dict,
                 model_name: str,
                 confidence_threshold: float = 0.5,
                 nms_Threshold: float = 0.4,
                 frame_size: Tuple[int, int] = [320, 320],
                 model_weights_url: str = "https://onedrive.live.com/download?cid=1D0DF2C7923ADAA7&resid=1D0DF2C7923ADAA7%21385&authkey=AI8Kqsf-smehalo",
                 model_cfg_url: str = "https://onedrive.live.com/embed?cid=1D0DF2C7923ADAA7&resid=1D0DF2C7923ADAA7%21384&authkey=AJCNFbRpE_T06uA"):
        super().__init__(object_map=objects_map)
        self.confidenceThreshold = confidence_threshold
        self.nmsThreshold = nms_Threshold
        self.frame_size = tuple(frame_size)
        download_url(model_cfg_url, PATH_MODEL + "/" + model_name + ".cfg")
        download_url(model_weights_url, PATH_MODEL + "/" + model_name + ".weights")
        self.model = cv2.dnn.readNet(PATH_MODEL + "/" + model_name + ".weights",
                                         PATH_MODEL + "/" + model_name + ".cfg")

        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.image_shape)
        img = np.expand_dims(img, axis=0)
        return img

    def detect_with_desc(self, img: np.ndarray) -> List[Detection]:
        img_shape = img.shape[:2]

        #0.00392 = 1/255
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=self.frame_size, mean=(0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.output_layers)
        detections = []
        boxes = []
        confidences = []
        class_list_id = []

        for output in outs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > self.confidenceThreshold and class_id in self.object_map.keys():
                    center_x = int(detect[0] * img_shape[1])
                    center_y = int(detect[1] * img_shape[0])
                    w = int(detect[2]*img_shape[1])
                    h = int(detect[3] * img_shape[0])
                    boxes.append([int(center_x-w/2), int(center_y-h/2), w, h])
                    confidences.append(float(conf))
                    class_list_id.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidenceThreshold, self.nmsThreshold)

        for i in indices:
            j = i[0]
            box = boxes[j]
            bb = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            detections.append(Detection(bb, self.object_map[class_list_id[j]], confidences[j]))
        return detections
