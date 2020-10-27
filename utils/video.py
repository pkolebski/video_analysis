import cv2

from detectors.detector import BaseDetector


class Video:
    def __init__(self, video_path: str, detector: BaseDetector):
        self.capture = cv2.VideoCapture(video_path)
        self.detector = detector

    def analyze(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while self.capture.isOpened():
            _, frame = self.capture.read()
            detections = self.detector.detect_with_desc(frame)
            for detect in detections:
                frame = cv2.rectangle(
                    frame,
                    detect[0:2],
                    detect[2:4],
                    color=(250, 0, 0),
                    thickness=2
                )

                frame = cv2.putText(frame,
                                    detect[4],
                                    (int(detect[0])+10, int(detect[1])+15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 0, 0),
                                    2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()
