import cv2
from detectors.detector import BaseDetector
from trackers.iou import BaseTracker
import time
import sys
import os
import numpy as np

IMAGE_SUFFIX = ".jpg"


class Video:
    def __init__(self, video_path: str, detector: BaseDetector, tracker: BaseTracker, output_folder=None):
        self.capture = cv2.VideoCapture(video_path)
        self.detector = detector
        self.tracker = tracker
        self.output_folder = output_folder
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

    def analyze(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_id = 0
        frames_fps = []
        while self.capture.isOpened():
            start_time = time.time()
            _, frame = self.capture.read()
            if frame is None:
                break
            detections = self.detector.detect_with_desc(frame)
            if len(detections)>0:
                if self.tracker is not None:
                    self.tracker.match_bbs(detections, frame)
                    frame = self.tracker.plot_history(frame)
                for detect in detections:
                    frame = cv2.rectangle(
                        frame,
                        tuple([int(x) for x in detect.position[0:2]]),
                        tuple([int(x) for x in detect.position[2:4]]),
                        color=(250, 0, 0),
                        thickness=2,
                    )

                    frame = cv2.putText(frame,
                                        str(detect.obj_type),
                                        (int(detect.position[0]) + 10, int(detect.position[1]) + 15),
                                        font,
                                        0.5,
                                        (255, 0, 0),
                                        2)


            fps = "{:.3}".format(1/(time.time()-start_time))
            sys.stdout.write("\rfps " + fps)
            frames_fps.append(fps)
            if self.output_folder is not None:
                self.save_frame(frame=frame,
                                filename=self.output_folder + "/" + str(frame_id).zfill(5) + IMAGE_SUFFIX,
                                scale_percent=100)
            else:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_id = frame_id+1

        if self.output_folder is not None:
            f = open(os.path.join(self.output_folder, "fps.txt"), 'w')
            s1 = '\n'.join(frames_fps)
            f.write(s1)
            f.close()

        self.capture.release()
        cv2.destroyAllWindows()

    def save_frame(self, frame, filename, scale_percent):
        if scale_percent!= 100:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(filename, resized)
        else:
            cv2.imwrite(filename, frame)


def save_imgs_as_video(folder_path: str, video_name: str, suffix: str=IMAGE_SUFFIX, description: str=""):
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    if os.path.isfile(os.path.join(folder_path, "fps.txt")):
        f = open(os.path.join(folder_path, "fps.txt"), 'r')
        fps = f.readlines()
        f.close()

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    mean_frame = np.mean(frame)
    if mean_frame > 100:
        text_color = (0, 0, 0)
    else:
        text_color = (255, 255, 255)
    for index, image in enumerate(images):
        img = cv2.imread(os.path.join(folder_path, image))
        text = ""
        if fps is not None:
            text = fps[index][:-1]+"fps".ljust(7)
        text = text + description.replace("_", " ")
        frame = cv2.putText(img,
                            text,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.8,
                            text_color,
                            2)
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        video.write(frame)
    video.release()

