from utils.video import Video
from detectors.detector import ExampleDetector
from detectors.detector_ssd_mobile import DetectorSSDMobileNetv2
from detectors.yolo_v3 import Yolov3
from trackers.iou import IouTracker

# detector = Yolov3(model_weights="https://pjreddie.com/media/files/yolov3.weights",
#                   model_cfg="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
detector = DetectorSSDMobileNetv2()
tracker = IouTracker()
video = Video('data/sherbrooke_video.avi', detector, tracker)
video.analyze()
