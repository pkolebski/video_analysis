import cv2
from detectors.detector import BaseDetector
from trackers.iou import BaseTracker


class Video:
    def __init__(self, video_path: str, detector: BaseDetector, tracker: BaseTracker):
        self.capture = cv2.VideoCapture(video_path)
        self.detector = detector
        self.tracker = tracker

    def analyze(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while self.capture.isOpened():
            _, frame = self.capture.read()
            detections = self.detector.detect_with_desc(frame)
            self.tracker.match_bbs(detections)
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
                                    (int(detect.position[0])+10, int(detect.position[1])+15),
                                    font,
                                    0.5,
                                    (255, 0, 0),
                                    2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()
