import cv2
import numpy as np
import requests
import base64
import os
from typing import Union, Optional

# Import Mixins
from .geometry import GeometryMixin
from .color import ColorMixin
from .filters import FilterMixin
from .features import FeatureMixin
from .draw import DrawMixin
from .analysis import AnalysisMixin

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from PIL import Image, ImageGrab
except ImportError:
    Image = None

class Imazing(GeometryMixin, ColorMixin, FilterMixin, FeatureMixin, DrawMixin, AnalysisMixin):
    """
    Imazing Core: Inherits from all Mixins to provide a unified API.
    """

    def __init__(self, source: Union[str, np.ndarray, bytes, None] = None):
        self.image: Optional[np.ndarray] = None
        self.metadata = {}

        if source is not None:
            self.load(source)

    # --- I/O Operations ---

    def load(self, source: Union[str, np.ndarray, bytes]):
        """Smart loader that detects source type (Path, URL, Bytes, Array)."""
        if isinstance(source, np.ndarray):
            self.image = source.copy()
        elif isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                self._load_from_url(source)
            elif os.path.isfile(source):
                # Handle unicode paths via numpy
                stream = open(source, "rb")
                bytes_data = bytearray(stream.read())
                array = np.asarray(bytes_data, dtype=np.uint8)
                self.image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
                stream.close()
            elif source.startswith('data:image'):
                self._load_from_base64(source)
        elif isinstance(source, bytes):
            self._load_from_bytes(source)

        if self.image is None:
            raise ValueError("Could not load image from source.")
        return self

    def _load_from_url(self, url):
        try:
            resp = requests.get(url, stream=True).raw
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            self.image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error loading URL: {e}")

    def _load_from_bytes(self, data):
        arr = np.frombuffer(data, np.uint8)
        self.image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    def _load_from_base64(self, base64_str):
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        decoded_data = base64.b64decode(base64_str)
        self._load_from_bytes(decoded_data)

    @classmethod
    def capture_screen(cls, region=None):
        """Captures full screen or region (x, y, w, h)."""
        if not pyautogui: raise ImportError("pyautogui not installed")
        img = pyautogui.screenshot(region=region)
        return cls(np.array(img)[:, :, ::-1]) # RGB to BGR

    @classmethod
    def from_clipboard(cls):
        """Reads image from clipboard."""
        if not Image: raise ImportError("Pillow not installed")
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            return cls(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return None

    @classmethod
    def capture_webcam(cls, camera_index=0):
        """Captures a single frame from webcam."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened(): return None
        ret, frame = cap.read()
        cap.release()
        if ret: return cls(frame)
        return None

    def save(self, path, quality=95):
        """Saves image to disk. Handles format params automatically."""
        if self.image is None: return
        ext = os.path.splitext(path)[1].lower()
        params = []
        if ext in ['.jpg', '.jpeg']:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == '.png':
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9] # Max compression
        elif ext == '.webp':
            params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

        # Use imencode/write for unicode path support
        success, buffer = cv2.imencode(ext, self.image, params)
        if success:
            with open(path, "wb") as f:
                f.write(buffer)

    def to_base64(self, format='.jpg') -> str:
        """Returns Data URI base64 string."""
        _, buffer = cv2.imencode(format, self.image)
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format[1:]};base64,{b64}"

    def to_numpy(self):
        return self.image.copy()

    def show(self, window_name="Imazing Preview", wait=True):
        """Displays the image in a GUI window."""
        if self.image is None: return
        cv2.imshow(window_name, self.image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
