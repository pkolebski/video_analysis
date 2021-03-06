from utils.video import Video
import yaml
import importlib


#import tensorflow as tf
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)

with open('configs/Yolov4_tiny_mio_tcd.yaml') as file:
  config = yaml.safe_load(file)

module_name, class_name = config['model']['class'].rsplit(".", 1)
class_detector = getattr(importlib.import_module(module_name), class_name)
detector = class_detector(**config['model']['parameters'])

video = Video('data/rouen_video.avi', detector)
video.analyze()
