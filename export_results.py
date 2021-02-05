from utils.video import Video, save_imgs_as_video
import yaml
import importlib
from trackers.iou import IouTracker
from trackers.deepSORT import DeepSORT
import shutil
import os

TMP_FOLDER = "tmp"
EXPORT_FOLDER = "output"
configs = ['configs/Yolov4_tiny_mio_tcd.yaml']#,
configs = ['configs/Yolov4.yaml']
# configs = ['configs/SSD_mobile.yaml']

video_paths = [
    'data/koronaRondo.MP4',
    'data/1280x720.m4v',
    'data/rouen_video.avi',
    'data/stmarc_video.avi',
    'data/koronaWidokzGory.MP4',
    'data/sherbrooke_video.avi',
    'data/janaPawlaPoczatekRuchomy.MP4',
    'data/janaPawlaSokolnicza.MP4']

trackers = [IouTracker, DeepSORT]

for path in video_paths+configs:
    if not os.path.exists(path):
        print(path , " not found")


for video_path in video_paths:
    print(video_path)
    for config_file in configs:
        for tracker_class in trackers:
            with open(config_file) as file:
                config = yaml.safe_load(file)
            video_name = video_path.split("/")[-1].split(".")[0]
            if not os.path.exists(os.path.join(EXPORT_FOLDER, video_name)):
                os.makedirs(os.path.join(EXPORT_FOLDER, video_name))

            module_name, class_name = config['model']['class'].rsplit(".", 1)
            class_detector = getattr(importlib.import_module(module_name), class_name)
            detector = class_detector(**config['model']['parameters'])
            tracker = tracker_class()
            model_name = config['model']['name']
            tracker_name = type(tracker).__name__
            video = Video(video_path=video_path,
                          detector=detector,
                          tracker=tracker,
                          output_folder=TMP_FOLDER)
            video.analyze()

            if not os.path.join(EXPORT_FOLDER, video_name):
                os.makedirs(os.path.join(EXPORT_FOLDER, video_name))

            video_export_path = os.path.join(EXPORT_FOLDER,
                                             video_name,
                                             model_name + "_" + tracker_name + ".mp4")
            save_imgs_as_video(TMP_FOLDER, video_export_path, description=model_name + " " + tracker_name)
            shutil.rmtree(TMP_FOLDER)
