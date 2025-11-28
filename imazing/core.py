import cv2
import numpy as np
import requests
import base64
import os
from typing import Union, Optional

# Import the Utils
from .utils import ensure_valid, validate_image, ImageError

# Import Mixin Stubs (We will populate these in Phase 2)
from .geometry import GeometryMixin
from .color import ColorMixin
from .filters import FilterMixin
from .features import FeatureMixin
from .draw import DrawMixin
from .analysis import AnalysisMixin
from .utilities import UtilityMixin  # New Mixin for "Crazy Good" utils

class Imazing(GeometryMixin, ColorMixin, FilterMixin, FeatureMixin, DrawMixin, AnalysisMixin, UtilityMixin):
    """
    Imazing Core: The State Manager.
    """

    def __init__(self, source: Union[str, np.ndarray, bytes, None] = None):
        self.image: Optional[np.ndarray] = None
        self.metadata = {}  # Store EXIF or other data here

        if source is not None:
            self.load(source)

    # --- Smart Loader ---

    def load(self, source: Union[str, np.ndarray, bytes]):
        """
        Smart Loader: Detects source type and loads the image.
        Supports: File Paths, URLs, Base64 strings, Raw Bytes, Numpy Arrays.
        """
        try:
            if isinstance(source, np.ndarray):
                self.image = source.copy()

            elif isinstance(source, str):
                # CASE 1: Web URL
                if source.startswith(('http://', 'https://')):
                    self._load_from_url(source)

                # CASE 2: Base64 Data URI
                elif source.startswith('data:image'):
                    self._load_from_base64(source)

                # CASE 3: Local File Path
                elif os.path.isfile(source):
                    self._load_from_file(source)

                else:
                    # Fallback: Check if it's a raw base64 string (no header)
                    try:
                        self._load_from_base64(source)
                    except:
                        raise ImageError(f"File not found or invalid input: {source}")

            elif isinstance(source, bytes):
                self._load_from_bytes(source)

            else:
                raise ImageError(f"Unsupported input type: {type(source)}")

            # Final Validation
            validate_image(self.image)

        except Exception as e:
            raise ImageError(f"Failed to load image: {str(e)}")

        return self

    # --- Internal Loaders ---

    def _load_from_file(self, path):
        # Use numpy fromfile to handle Unicode paths on Windows correctly
        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        array = np.asarray(bytes_data, dtype=np.uint8)
        self.image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        stream.close()

    def _load_from_url(self, url):
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            self.image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise ImageError(f"URL Error: {e}")

    def _load_from_bytes(self, data):
        arr = np.frombuffer(data, np.uint8)
        self.image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    def _load_from_base64(self, base64_str):
        # Clean the string if it contains headers
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        decoded_data = base64.b64decode(base64_str)
        self._load_from_bytes(decoded_data)
    
    @classmethod
    def capture_screen(cls, region=None):
        """Captures full screen or region (x, y, w, h)."""
        import pyautogui
        # pyautogui returns RGB, we need BGR for OpenCV
        img = pyautogui.screenshot(region=region)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return cls(img_bgr)

    @classmethod
    def from_clipboard(cls):
        """Reads image from clipboard."""
        from PIL import ImageGrab
        img = ImageGrab.grabclipboard()
        if img is not None and hasattr(img, 'convert'):
            img_np = np.array(img.convert('RGB'))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return cls(img_bgr)
        return None

    @classmethod
    def capture_webcam(cls, camera_index=0):
        """Captures a single frame from webcam."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cls(frame)
        return None

    # --- Core Output ---

    @ensure_valid
    def save(self, path, quality=95):
        """Saves image to disk with format-specific optimization."""
        ext = os.path.splitext(path)[1].lower()
        params = []

        if ext in ['.jpg', '.jpeg']:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == '.png':
            # Map quality (0-100) to PNG compression (0-9)
            compression = int((100 - quality) / 10.0)
            compression = max(0, min(9, compression))
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]
        elif ext == '.webp':
            params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

        success, buffer = cv2.imencode(ext, self.image, params)
        if success:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            raise ImageError("Failed to encode image for saving.")
        return self

    @ensure_valid
    def show(self, window_name="Imazing Preview", wait=True):
        """Displays the image."""
        cv2.imshow(window_name, self.image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return self

    @ensure_valid
    def to_numpy(self):
        """Returns a copy of the underlying numpy array."""
        return self.image.copy()

    @ensure_valid
    def to_base64(self, format='.jpg') -> str:
        """Returns Data URI base64 string."""
        _, buffer = cv2.imencode(format, self.image)
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format[1:]};base64,{b64}"

    @classmethod
    def load_batch(cls, folder_path, extensions=['.jpg', '.png', '.jpeg']):
        """
        Loads all images from a folder.
        Returns a list of Imazing objects.
        """
        images = []
        if not os.path.exists(folder_path):
            return images

        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                try:
                    full_path = os.path.join(folder_path, filename)
                    img = cls(full_path)
                    images.append(img)
                except:
                    continue
        return images
