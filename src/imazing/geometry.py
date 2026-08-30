from __future__ import annotations

import cv2
import numpy as np
import random
import warnings

from ._validation import requires_image


class GeometryMixin:
    """Handles Shape, Size, and Orientation"""

    @requires_image
    def resize(self, width=None, height=None, inter=None):
        """Resizes while optionally preserving aspect ratio.

        `inter` defaults to an automatic choice based on the resize
        direction: INTER_AREA (best for shrinking, per OpenCV's own
        guidance) when the result is smaller than the original, INTER_LINEAR
        (better quality than area-averaging for enlarging) when it's
        larger. Pass an explicit `inter=` to override the automatic choice.
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

        if inter is None:
            is_upscale = (dim[0] * dim[1]) > (w * h)
            inter = cv2.INTER_LINEAR if is_upscale else cv2.INTER_AREA

        self.image = cv2.resize(self.image, dim, interpolation=inter)
        return self

    @requires_image
    def crop(self, x, y, w, h):
        """Crops a region.

        Returns an independent copy rather than a view into the original
        array, so the source image's full buffer isn't kept alive in memory
        just because a small crop of it is still referenced.
        """
        self.image = self.image[y:y+h, x:x+w].copy()
        return self

    @requires_image
    def rotate(self, angle, scale=1.0):
        """Rotates image around center."""
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))
        return self

    @requires_image
    def flip(self, horizontal=True, vertical=False):
        flip_code = -1 if horizontal and vertical else (1 if horizontal else 0)
        self.image = cv2.flip(self.image, flip_code)
        return self

    @requires_image
    def pad(self, top, bottom, left, right, color=(0, 0, 0)):
        """Adds border/padding."""
        self.image = cv2.copyMakeBorder(
            self.image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color
        )
        return self

    @requires_image
    def warp_perspective(self, src_points, dst_points, size):
        """Applies 4-point transform."""
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.image = cv2.warpPerspective(self.image, M, size)
        return self

    def augment_random(self):
        """Applies random augmentations for ML datasets."""
        _AUGMENTATION_TYPES = {
            'flip' : ("flip", self.flip, {"horizontal" : random.choice([True, False]), "vertical" : random.choice([True, False])}),
            'rotate' : ("rotate", self.rotate, {"angle" : random.uniform(-15, 15)}),
            'noise' : ("adjust_brightness_contrast", self.adjust_brightness_contrast, {"alpha" : 1.0, "beta" : random.uniform(-30, 30)}),
            'brightness' : ("add_noise", self.add_noise, {})
        }
        _available_types = []
        for type in _AUGMENTATION_TYPES.keys() :
            if hasattr(self, _AUGMENTATION_TYPES.get(type, ["__none"])[0]) :
                _available_types.append(type)
            else :
                warnings.warn(
                    f"{type} augmentation is unavailable because "
                    "'adjust_brightness_contrast' is not implemented.",
                    UserWarning
                )

        choice = random.choice(_available_types)

        _, _method, _params = _AUGMENTATION_TYPES.get(choice)
        _method(**_params)

        return self