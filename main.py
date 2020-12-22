from utils.video import Video, save_imgs_as_video
import yaml
import importlib
from trackers.iou import IouTracker
from trackers.deepSORT import DeepSORT


with open('configs/Yolov4_mio_tcd.yaml') as file:
  config = yaml.safe_load(file)
module_name, class_name = config['model']['class'].rsplit(".", 1)
class_detector = getattr(importlib.import_module(module_name), class_name)
detector = class_detector(**config['model']['parameters'])
tracker = DeepSORT()
model_name = config['model']['parameters']['model_name']
tracker_name = type(tracker).__name__

video = Video(video_path='data/custom_data/koronaRondo.MP4',
              detector=detector,
              tracker=tracker)
video.analyze()