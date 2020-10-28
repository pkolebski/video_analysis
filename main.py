from utils.video import Video
from detectors.detector import ExampleDetector
from detectors.detector_ssd_mobile import DetectorSSDMobileNetv2
from detectors.yolo_v3 import Yolov3

detector = Yolov3()
video = Video('data/sherbrooke_video.avi', detector)
video.analyze()
