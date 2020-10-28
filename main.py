from utils.video import Video
from detectors.detector import ExampleDetector
from detectors.detector_ssd_mobile import DetectorSSDMobileNetv2
from detectors.yolo_v3 import Yolov3

detector = Yolov3(threshold=0.1,
                  model_weights="https://pjreddie.com/media/files/yolov3.weights",
                  model_cfg="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
video = Video('data/sherbrooke_video.avi', detector)
video.analyze()
