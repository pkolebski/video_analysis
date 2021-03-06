import cv2
from detectors.detector import BaseDetector
import time
import sys


class Video:
    def __init__(self, video_path: str, detector: BaseDetector):
        self.capture = cv2.VideoCapture(video_path)
        self.detector = detector

    def analyze(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while self.capture.isOpened():
            start_time = time.time()
            _, frame = self.capture.read()
            detections = self.detector.detect_with_desc(frame)
            for detect in detections:
                frame = cv2.rectangle(
                    frame,
                    detect.position[0:2],
                    detect.position[2:4],
                    color=(250, 0, 0),
                    thickness=2
                )

                frame = cv2.putText(frame,
                                    str(detect.obj_type),
                                    (int(detect.position[0]) + 10, int(detect.position[1]) + 15),
                                    font,
                                    0.5,
                                    (255, 0, 0),
                                    2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            sys.stdout.write("\rfps {:.5}".format(1/(time.time()-start_time)))
        self.capture.release()
        cv2.destroyAllWindows()
