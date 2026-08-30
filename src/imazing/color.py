import cv2
import numpy as np

from ._validation import requires_image, convert_to, VALID_COLOR_SPACES


class ColorMixin:
    """Handles Colors, Tone, and Brightness"""

    @requires_image
    def convert_color(self, mode='GRAY'):
        """Converts color space. Options: GRAY, HSV, RGB, LAB, BGR, BGRA.

        Converts from the image's *actual current* color space.
        """
        mode = mode.upper()
        if mode not in VALID_COLOR_SPACES:
            raise ValueError(
                f"Invalid colour mode '{mode}'. "
                f"Valid modes are: {','.join(VALID_COLOR_SPACES)}."
            )

        current = self.color_space or 'BGR'
        if mode != current:
            self.image = convert_to(self.image, current, mode)
            self.color_space = mode
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
        """Improves contrast. Works on any color space (not just BGR) by
        equalizing luminance via a BGR/YUV round-trip, and leaves an
        existing alpha channel untouched rather than equalizing it too."""
        if self.image.ndim == 2:
            self.image = cv2.equalizeHist(self.image)
            return self

        current = self.color_space or 'BGR'
        image = self.image
        alpha = None
        color_space = current
        if current == 'BGRA':
            image, alpha = image[:, :, :3], image[:, :, 3]
            color_space = 'BGR'  # BGRA's first 3 channels are plain BGR

        # Convert to BGR, equalize the Y channel via YUV, convert back.
        yuv = convert_to(image, color_space, 'YUV')
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        result = convert_to(yuv, 'YUV', color_space)

        if alpha is not None:
            result = np.dstack([result, alpha])

        self.image = result
        return self