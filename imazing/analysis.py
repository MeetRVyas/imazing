import cv2
import numpy as np

from ._validation import requires_image


class AnalysisMixin:
    """Handles Statistics and Metadata"""

    @requires_image
    def get_stats(self):
        """Returns basic statistics."""
        return {
            "width": self.image.shape[1],
            "height": self.image.shape[0],
            "channels": self.image.shape[2] if len(self.image.shape) > 2 else 1,
            "mean": np.mean(self.image),
            "std": np.std(self.image),
            "min": np.min(self.image),
            "max": np.max(self.image)
        }

    @requires_image
    def compute_hash(self, size=8):
        """Perceptive hash for duplicate detection (aHash)."""
        resized = cv2.resize(self.image, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        diff = gray > avg
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])