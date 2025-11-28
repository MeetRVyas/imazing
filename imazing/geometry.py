import cv2
import numpy as np
import random
from .utils import ensure_valid

class GeometryMixin:
    """Handles Shape, Size, Orientation, and Smart Transforms."""
    
    @ensure_valid
    def resize(self, width=None, height=None, inter=None):
        """
        Resizes image using Adaptive Interpolation.
        Upscale -> INTER_LANCZOS4 (Best Quality)
        Downscale -> INTER_AREA (Best Moire-free)
        """
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

        # Logic: Auto-select interpolation if not provided
        if inter is None:
            if dim[0] > w or dim[1] > h:
                inter = cv2.INTER_LANCZOS4 # Best for Upscaling
            else:
                inter = cv2.INTER_AREA     # Best for Downscaling

        self.image = cv2.resize(self.image, dim, interpolation=inter)
        return self
    
    @ensure_valid
    def crop(self, x, y, w, h):
        """Crops a rectangular region."""
        img_h, img_w = self.image.shape[:2]
        # Ensure crop is within bounds
        x, y = max(0, x), max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        self.image = self.image[y:y+h, x:x+w]
        return self

    @ensure_valid
    def rotate(self, angle, scale=1.0, expand=False):
        """
        Rotates image.
        :param expand: If True, resizes canvas so corners aren't cut off (Smart Rotation).
        """
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        if expand:
            # Calculate new bounding box dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust the translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            self.image = cv2.warpAffine(self.image, M, (new_w, new_h))
        else:
            self.image = cv2.warpAffine(self.image, M, (w, h))

        return self

    @ensure_valid
    def flip(self, horizontal=True, vertical=False):
        flip_code = -1 if horizontal and vertical else (1 if horizontal else 0)
        self.image = cv2.flip(self.image, flip_code)
        return self

    @ensure_valid
    def pad(self, top, bottom, left, right, color=(0, 0, 0)):
        """Adds border/padding."""
        self.image = cv2.copyMakeBorder(self.image, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=color)
        return self

    @ensure_valid
    def warp_perspective(self, src_points, dst_points, size):
        """Applies 4-point transform."""
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.image = cv2.warpPerspective(self.image, M, size)
        return self

    @ensure_valid
    def augment_random(self):
        """Applies random geometric augmentations."""
        choice = random.choice(['flip', 'rotate', 'zoom', 'shift'])

        if choice == 'flip':
            self.flip(random.choice([True, False]), random.choice([True, False]))
        elif choice == 'rotate':
            # Rotate with expansion to keep data
            self.rotate(random.uniform(-20, 20), expand=True)
        elif choice == 'zoom':
            # Simple center zoom crop
            h, w = self.image.shape[:2]
            factor = random.uniform(0.8, 0.95)
            new_w, new_h = int(w*factor), int(h*factor)
            x, y = (w-new_w)//2, (h-new_h)//2
            self.crop(x, y, new_w, new_h)
            self.resize(w, h) # Resize back to original
        return self


    @ensure_valid
    def affine_transform(self, matrix, size=None):
        """Applies a raw 2x3 Affine Matrix."""
        h, w = self.image.shape[:2]
        if size is None: size = (w, h)
        self.image = cv2.warpAffine(self.image, matrix, size)
        return self

    @ensure_valid
    def warp_arbitrary(self, map_x, map_y):
        """Applies pixel-level warping using remap maps."""
        self.image = cv2.remap(self.image, map_x, map_y, cv2.INTER_LINEAR)
        return self

    @ensure_valid
    def splice_region(self, region_img, x, y):
        """Pastes a numpy region into the image at (x,y)."""
        h, w = region_img.shape[:2]
        self.image[y:y+h, x:x+w] = region_img
        return self

    @ensure_valid
    def auto_orient(self, orientation=None):
        """
        Fixes orientation based on EXIF tag (1-8).
        If orientation is None, tries to read from self.metadata.
        """
        if orientation is None:
            orientation = self.metadata.get('Orientation', 1)
            # Try to cast to int if it's a string from EXIF dict
            try: orientation = int(orientation)
            except: orientation = 1

        if orientation == 3:
            self.image = cv2.rotate(self.image, cv2.ROTATE_180)
        elif orientation == 6:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self

    @ensure_valid
    def augment_crop_pad(self, pad=20, crop_factor=0.9):
        """
        Pads the image (reflection) then randomly crops back to original size.
        Standard ML augmentation technique.
        """
        h, w = self.image.shape[:2]
        # Pad
        self.image = cv2.copyMakeBorder(self.image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        # Random Crop
        new_h, new_w = self.image.shape[:2]
        x = random.randint(0, new_w - w)
        y = random.randint(0, new_h - h)
        self.image = self.image[y:y+h, x:x+w]
        return self
