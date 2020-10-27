from utils.video import Video
from detectors.detector import ExampleDetector
from detectors.detector_ssd_mobile import DetectorSSDMobileNetv2

detector = DetectorSSDMobileNetv2()
video = Video('data/sherbrooke_video.avi', detector)
video.analyze()
