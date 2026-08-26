import cv2
import numpy as np

from ._validation import requires_image


_VALID_COLOR_MODES = {
    'GRAY' : cv2.COLOR_BGR2GRAY,
    'HSV' : cv2.COLOR_BGR2HSV,
    'RGB' : cv2.COLOR_BGR2RGB,
    'LAB' : cv2.COLOR_BGR2Lab,
    'BGR' : cv2.COLOR_GRAY2BGR
}


class ColorMixin:
    """Handles Colors, Tone, and Brightness"""

    @requires_image
    def convert_color(self, mode='GRAY'):
        """Converts color space. Options: GRAY, HSV, RGB, LAB, BGR."""
        mode = mode.upper()
        if mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"Invalid colour mode '{mode}'. "
                f"Valid modes are: {','.join(_VALID_COLOR_MODES)}."
            )

        elif mode != 'GRAY' or len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, _VALID_COLOR_MODES.get(mode))
        return self

    @requires_image
    def adjust_brightness_contrast(self, alpha=1.0, beta=0):
        """alpha: contrast (1.0-3.0), beta: brightness (0-100)."""
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self

    @requires_image
    def invert(self):
        self.image = cv2.bitwise_not(self.image)
        return self

    @requires_image
    def histogram_equalization(self):
        """Improves contrast (works best on grayscale or per channel)."""
        if len(self.image.shape) == 2:
            self.image = cv2.equalizeHist(self.image)
        else:
            # Convert to YUV, equalize Y, convert back
            yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            self.image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return self