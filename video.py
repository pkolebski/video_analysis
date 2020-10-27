import cv2

from detector import Detector


class Video:
    def __init__(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        self.detector = Detector()

    def analyze(self):
        while self.capture.isOpened():
            _, frame = self.capture.read()
            print(type(frame))
            detections = self.detector.detect(frame)
            for detect in detections:
                frame = cv2.rectangle(
                    frame,
                    detect[:2],
                    detect[2:],
                    color=(255, 0, 0),
                    thickness=3,
                )

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()
