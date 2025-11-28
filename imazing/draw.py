import cv2
import numpy as np
from .utils import ensure_valid

class DrawMixin:
    """Handles Annotation, Shapes, Censors, and Fun Drawings."""

    @ensure_valid
    def draw_rect(self, x, y, w, h, color=(0, 255, 0), thickness=2):
        cv2.rectangle(self.image, (x, y), (x+w, y+h), color, thickness)
        return self

    @ensure_valid
    def draw_circle(self, x, y, r, color=(0, 0, 255), thickness=-1):
        cv2.circle(self.image, (x, y), r, color, thickness)
        return self

    @ensure_valid
    def draw_line(self, x1, y1, x2, y2, color=(0,255,0), thickness=2):
        cv2.line(self.image, (x1, y1), (x2, y2), color, thickness)
        return self

    @ensure_valid
    def draw_text(self, text, x, y, size=1.0, color=(255, 255, 255), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.putText(self.image, text, (x, y), font, size, color, thickness, cv2.LINE_AA)
        return self

    @ensure_valid
    def overlay_image(self, overlay_img, x, y, alpha=0.5):
        """Overlays another numpy array with transparency."""
        h, w = overlay_img.shape[:2]
        img_h, img_w = self.image.shape[:2]

        # Check bounds
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
             # Basic safety: just skip if out of bounds to prevent crash
             # Or crop overlay to fit
             return self 

        roi = self.image[y:y+h, x:x+w]

        # If overlay has alpha channel (4 channels)
        if overlay_img.shape[2] == 4:
            alpha_mask = overlay_img[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask

            for c in range(3):
                roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] + 
                                alpha_inv * roi[:, :, c])
            self.image[y:y+h, x:x+w] = roi
        else:
            blended = cv2.addWeighted(roi, 1 - alpha, overlay_img, alpha, 0)
            self.image[y:y+h, x:x+w] = blended

        return self

    # --- Fun / Advanced Annotations ---

    @ensure_valid
    def draw_pixelate(self, x, y, w, h, blocks=10):
        """Applies 'Censor' mosaic effect to a region."""
        roi = self.image[y:y+h, x:x+w]
        # Shrink
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        # Scale back up (Nearest Neighbor makes it pixelated)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        self.image[y:y+h, x:x+w] = pixelated
        return self

    @ensure_valid
    def draw_speech_bubble(self, text, x, y, w=200, h=100, color=(255, 255, 255), text_color=(0,0,0)):
        """Draws a comic-style speech bubble pointing to (x,y)."""
        # Bubble body (ellipse) above the point
        center_x = x + 50
        center_y = y - h - 20
        axes = (w // 2, h // 2)

        cv2.ellipse(self.image, (center_x, center_y), axes, 0, 0, 360, color, -1)

        # Tail (Triangle)
        pts = np.array([[x, y], [center_x - 20, center_y + h//2 - 10], [center_x + 20, center_y + h//2 - 10]], np.int32)
        cv2.fillPoly(self.image, [pts], color)

        # Text inside
        self.draw_text(text, center_x - w//3, center_y, size=0.6, color=text_color, thickness=1)
        return self

    @ensure_valid
    def draw_fancy_arrow(self, x1, y1, x2, y2, color=(0, 0, 255), thickness=3, tip_len=0.3):
        """Draws a thick, bold arrow."""
        cv2.arrowedLine(self.image, (x1, y1), (x2, y2), color, thickness, line_type=cv2.LINE_AA, tipLength=tip_len)
        return self

    @ensure_valid
    def draw_progress_bar(self, percent, x=20, y=None, w=None, h=20, color=(0, 255, 0)):
        """Draws a visual progress bar (e.g., for video processing)."""
        img_h, img_w = self.image.shape[:2]
        if y is None: y = img_h - 40
        if w is None: w = img_w - 40

        # Background
        cv2.rectangle(self.image, (x, y), (x+w, y+h), (50, 50, 50), -1)
        # Fill
        fill_w = int(w * (percent / 100.0))
        cv2.rectangle(self.image, (x, y), (x+fill_w, y+h), color, -1)
        # Border
        cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 255, 255), 1)
        return self

    @ensure_valid
    def draw_callout(self, text, x, y, point_x, point_y, color=(0, 255, 0)):
        """
        Draws a text box at (x,y) with a line pointing to (point_x, point_y).
        Useful for labeling objects in technical diagrams.
        """
        # Draw the target point
        cv2.circle(self.image, (point_x, point_y), 3, color, -1)

        # Calculate text size
        (fw, fh), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw box background
        padding = 5
        cv2.rectangle(self.image, (x, y - fh - padding), (x + fw + padding*2, y + padding), (0,0,0), -1)
        cv2.rectangle(self.image, (x, y - fh - padding), (x + fw + padding*2, y + padding), color, 1)

        # Draw Text
        cv2.putText(self.image, text, (x + padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Draw Line connecting box to point
        # determine closest edge of box
        box_center_x = x + fw//2
        box_center_y = y - fh//2
        cv2.line(self.image, (box_center_x, box_center_y + fh), (point_x, point_y), color, 1)
        return self


    @ensure_valid
    def draw_polygon(self, points, color=(0, 255, 0), thickness=2, closed=True):
        """Draws a polygon defined by a list of (x,y) tuples."""
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.image, [pts], closed, color, thickness)
        return self
