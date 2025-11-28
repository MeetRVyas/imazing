import cv2
from .core import Imazing

class VideoStream:
    """Helper for reading/writing video or continuous webcam processing."""

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = None

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return Imazing(frame)
        return None

    def start_recording(self, filename, fps=None):
        if fps is None: fps = self.fps if self.fps > 0 else 20.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))

    def write_frame(self, imazing_obj):
        if self.writer:
            self.writer.write(imazing_obj.image)

    def release(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
