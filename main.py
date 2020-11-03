from utils.video import Video
from detectors.detector import ExampleDetector
from detectors.detector_ssd_mobile import DetectorSSDMobileNetv2
from detectors.yolo_v3 import Yolov3, OBJECTS_MAP_MIO_TCD

detector = Yolov3(objects_map=OBJECTS_MAP_MIO_TCD,
                  model_weights="https://onedrive.live.com/download?cid=1D0DF2C7923ADAA7&resid=1D0DF2C7923ADAA7%21385&authkey=AI8Kqsf-smehalo",
                  model_cfg="https://onedrive.live.com/download?cid=1D0DF2C7923ADAA7&resid=1D0DF2C7923ADAA7%21384&authkey=AJCNFbRpE_T06uA",
                  model_name="Yolov4_tiny_mio_tcd")
video = Video('data/1280x720.m4v', detector)
video.analyze()
