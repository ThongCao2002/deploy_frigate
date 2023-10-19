import threading
import cv2


class VideoDecoder:
    """
    Class that continously decodes frames from a VideoCapture object
    with a dedicated thread
    """

    def __init__(self, src=None):
        if src is None:
            src = 0

        self.stream = cv2.VideoCapture(src)
        self.decoded, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.decode, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def decode(self):
        while not self.stopped:
            if not self.decoded:
                self.stop()
            else:
                self.decoded, self.frame = self.stream.read()
