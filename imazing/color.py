import cv2
import numpy as np
from .utils import ensure_valid

class ColorMixin:
    """Handles Colors, Tone, Saturation, and Gamma."""

    @ensure_valid
    def convert_color(self, mode='GRAY'):
        """Converts color space. Options: GRAY, HSV, RGB, LAB, BGR."""
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

    @ensure_valid
    def adjust_brightness_contrast(self, alpha=1.0, beta=0):
        """
        alpha: contrast (1.0-3.0)
        beta: brightness (0-100)
        """
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self

    @ensure_valid
    def adjust_saturation(self, factor=1.0):
        """
        Adjust color saturation. 
        0.0 = Grayscale, 1.0 = Original, >1.0 = Vibrant
        """
        if len(self.image.shape) < 3: return self # Skip grayscale

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor # Scale Saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self

    @ensure_valid
    def adjust_gamma(self, gamma=1.0):
        """
        Non-linear brightness adjustment.
        gamma < 1.0 = brighter / gamma > 1.0 = darker
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        self.image = cv2.LUT(self.image, table)
        return self

    @ensure_valid
    def shift_hue(self, amount=0):
        """Shifts the Hue channel (0-180 in OpenCV)."""
        if len(self.image.shape) < 3: return self

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + amount) % 180
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self

    @ensure_valid
    def invert(self):
        self.image = cv2.bitwise_not(self.image)
        return self

    @ensure_valid
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

    @ensure_valid
    def extract_color_mask(self, lower_hsv, upper_hsv):
        """
        Returns a binary mask of pixels within color range.
        Updates self.image to be the mask.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        return self

    @ensure_valid
    def reduce_palette(self, k=8):
        """K-Means Color Quantization (Posterization)."""
        data = self.image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        self.image = result.reshape(self.image.shape)
        return self


    @ensure_valid
    def augment_color_random(self):
        """
        Randomly adjusts Hue, Saturation, Brightness, and Contrast.
        Great for ML Data Augmentation.
        """
        # Random parameters
        hue = random.randint(-10, 10)
        sat = random.uniform(0.5, 1.5)
        bright = random.randint(-30, 30)
        contrast = random.uniform(0.8, 1.2)

        # Apply Logic using existing helpers or direct manipulation
        # 1. HSV for Hue/Sat
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        hsv[:, :, 1] *= sat
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Brightness/Contrast
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=bright)

        self.image = img
        return self


    @ensure_valid
    def enhance_clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Contrast Limited Adaptive Histogram Equalization."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        res = clahe.apply(gray)
        if len(self.image.shape) == 3:
            # Apply only to Luminance channel if color
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            self.image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            self.image = res
        return self
