import cv2
import numpy as np
import random

class GeometryMixin:
    """Handles Shape, Size, and Orientation"""

    def resize(self, width=None, height=None, inter=cv2.INTER_AREA):
        """Resizes while optionally preserving aspect ratio."""
        h, w = self.image.shape[:2]
        if width is None and height is None: return self

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

        self.image = cv2.resize(self.image, dim, interpolation=inter)
        return self

    def crop(self, x, y, w, h):
        """Crops a region."""
        self.image = self.image[y:y+h, x:x+w]
        return self

    def rotate(self, angle, scale=1.0):
        """Rotates image around center."""
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))
        return self

    def flip(self, horizontal=True, vertical=False):
        flip_code = -1 if horizontal and vertical else (1 if horizontal else 0)
        self.image = cv2.flip(self.image, flip_code)
        return self

    def pad(self, top, bottom, left, right, color=(0, 0, 0)):
        """Adds border/padding."""
        self.image = cv2.copyMakeBorder(self.image, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=color)
        return self

    def warp_perspective(self, src_points, dst_points, size):
        """Applies 4-point transform."""
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.image = cv2.warpPerspective(self.image, M, size)
        return self

    def augment_random(self):
        """Applies random augmentations for ML datasets."""
        ops = ['flip', 'rotate', 'noise', 'brightness']
        choice = random.choice(ops)

        if choice == 'flip':
            self.flip(random.choice([True, False]), random.choice([True, False]))
        elif choice == 'rotate':
            self.rotate(random.uniform(-15, 15))
        elif choice == 'brightness':
            # Assumes ColorMixin is also present in the main class
            if hasattr(self, 'adjust_brightness_contrast'):
                self.adjust_brightness_contrast(alpha=1.0, beta=random.uniform(-30, 30))
        elif choice == 'noise':
             # Assumes FilterMixin is also present
            if hasattr(self, 'add_noise'):
                self.add_noise()
        return self
