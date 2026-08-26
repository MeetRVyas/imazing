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
    def overlay_image(self, overlay_img, x=0, y=0, alpha=0.5):
        """Overlays another image (Imazing object / numpy array) with transparency."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        if not isinstance(overlay_img, np.ndarray):
            overlay_img = overlay_img.to_numpy()
        
        h, w = overlay_img.shape[:2]
        img_h, img_w = self.image.shape[:2]

        # Intersect the overlay's bounding box (which may start negative or
        # extend past the edges) with the base image's bounds.
        dest_x0, dest_y0 = max(x, 0), max(y, 0)
        dest_x1, dest_y1 = min(x + w, img_w), min(y + h, img_h)

        if dest_x1 <= dest_x0 or dest_y1 <= dest_y0:
            return self  # overlay doesn't touch the image at all

        # The matching crop of the overlay itself,
        # offset by however much was cut off on the left/top.
        src_x0, src_y0 = dest_x0 - x, dest_y0 - y
        src_x1 = src_x0 + (dest_x1 - dest_x0)
        src_y1 = src_y0 + (dest_y1 - dest_y0)

        roi = self.image[dest_y0:dest_y1, dest_x0:dest_x1]
        overlay_crop = overlay_img[src_y0:src_y1, src_x0:src_x1]

        blended = cv2.addWeighted(roi, 1 - alpha, overlay_crop, alpha, 0)
        self.image[dest_y0:dest_y1, dest_x0:dest_x1] = blended
        return self