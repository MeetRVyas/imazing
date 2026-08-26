import cv2
import numpy as np

from ._validation import requires_image


class DrawMixin:
    """Handles Annotation and Overlays"""

    @requires_image
    def draw_rect(self, x, y, w, h, color=(0, 255, 0), thickness=2):
        cv2.rectangle(self.image, (x, y), (x+w, y+h), color, thickness)
        return self

    @requires_image
    def draw_circle(self, x, y, r, color=(0, 0, 255), thickness=-1):
        cv2.circle(self.image, (x, y), r, color, thickness)
        return self

    @requires_image
    def draw_text(self, text, x, y, size=1.0, color=(255, 255, 255), thickness=1):
        cv2.putText(self.image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        return self

    @requires_image
    def overlay_image(self, overlay_img, x = 0, y = 0, alpha = 0.5):
        """Overlays another image (numpy array) with transparency."""
        h, w = overlay_img.shape[:2]
        if x < 0 or y < 0 or y + h > self.image.shape[0] or x + w > self.image.shape[1]:
            # Simple boundary check: trim overlay if needed or just return
            raise ValueError(
                "Overlay image is outside the valid image boundaries."
            )
            return self

        roi = self.image[y:y+h, x:x+w]
        blended = cv2.addWeighted(roi, 1 - alpha, overlay_img, alpha, 0)
        self.image[y:y+h, x:x+w] = blended
        return self