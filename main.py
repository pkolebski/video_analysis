from utils.video import Video, save_imgs_as_video
import yaml
import importlib
from trackers.iou import IouTracker
from trackers.deepSORT import DeepSORT


with open('configs/SSD_mobile.yaml') as file:
  config = yaml.safe_load(file)
module_name, class_name = config['model']['class'].rsplit(".", 1)
class_detector = getattr(importlib.import_module(module_name), class_name)
detector = class_detector(**config['model']['parameters'])
tracker = DeepSORT()
model_name = config['model']['name']
tracker_name = type(tracker).__name__

video = Video(video_path='data/VID_20201118_110914.mp4',
              detector=detector,
              tracker=tracker,
              output_folder='out')
video.analyze()