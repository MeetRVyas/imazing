import cv2
import numpy as np

class ColorMixin:
    """Handles Colors, Tone, and Brightness"""

    def convert_color(self, mode='GRAY'):
        """Converts color space. Options: GRAY, HSV, RGB, LAB."""
        mode = mode.upper()
        if mode == 'GRAY' and len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif mode == 'HSV':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif mode == 'RGB':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif mode == 'LAB':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        elif mode == 'BGR' and len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        return self

    def adjust_brightness_contrast(self, alpha=1.0, beta=0):
        """alpha: contrast (1.0-3.0), beta: brightness (0-100)."""
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self

    def invert(self):
        self.image = cv2.bitwise_not(self.image)
        return self

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
