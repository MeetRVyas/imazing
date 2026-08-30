"""Shared helpers. Not part of the public API."""

from __future__ import annotations

import functools

import cv2

from .exceptions import NoImageLoadedError


def requires_image(method):
    """Guard a mixin method that operates on `self.image`.

    Without this, calling e.g. `Imazing().resize(width=10)` fails deep
    inside OpenCV with a bare `AttributeError: 'NoneType' object has no
    attribute 'shape'`.
    
    This raises a clear, catchable error at the point of the actual mistake instead."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.image is None:
            raise NoImageLoadedError(
                f"{method.__qualname__}() requires a loaded image, but none is set. "
                "Call .load(source) first, or pass a source to Imazing(source)."
            )
        return method(self, *args, **kwargs)
    return wrapper


# Color-space bookkeeping ------------------------------------------------
VALID_COLOR_SPACES = ('GRAY', 'BGR', 'RGB', 'HSV', 'LAB', 'BGRA', 'YUV')

# BGR is the hub: every space converts to/from BGR directly, and a
# transition between two non-BGR spaces (e.g. HSV -> LAB) is routed through
# it, since OpenCV doesn't expose most of those pairs as a single flag.
_TO_BGR = {
    'GRAY': cv2.COLOR_GRAY2BGR,
    'RGB': cv2.COLOR_RGB2BGR,
    'HSV': cv2.COLOR_HSV2BGR,
    'LAB': cv2.COLOR_Lab2BGR,
    'BGRA': cv2.COLOR_BGRA2BGR,
    'YUV': cv2.COLOR_YUV2BGR,
}

_FROM_BGR = {
    'GRAY': cv2.COLOR_BGR2GRAY,
    'RGB': cv2.COLOR_BGR2RGB,
    'HSV': cv2.COLOR_BGR2HSV,
    'LAB': cv2.COLOR_BGR2Lab,
    'BGRA': cv2.COLOR_BGR2BGRA,
    'YUV': cv2.COLOR_BGR2YUV,
}


def infer_color_space(image):
    """Best-effort guess of an array's color space from its shape alone, 
    used the moment an image is loaded.
    1/3/4-channel -> GRAY/BGR/BGRA"""
    if image is None:
        return None
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return 'GRAY'
    if image.ndim == 3 and image.shape[2] == 3:
        return 'BGR'
    if image.ndim == 3 and image.shape[2] == 4:
        return 'BGRA'
    return None  # unusual channel count -- caller shouldn't assume a space


def convert_to(image, current, target):
    """Converts `image` from `current` color space to `target`, routing
    through BGR when there's no direct OpenCV conversion code.
    Returns a new array; never mutates `image` in place."""
    current = (current or 'BGR').upper()
    target = target.upper()
    if current not in VALID_COLOR_SPACES:
        raise ValueError(f"Unknown source color space '{current}'.")
    if target not in VALID_COLOR_SPACES:
        raise ValueError(f"Unknown target color space '{target}'.")
    if current == target:
        return image
    if current != 'BGR':
        image = cv2.cvtColor(image, _TO_BGR[current])
    if target != 'BGR':
        image = cv2.cvtColor(image, _FROM_BGR[target])
    return image


def as_gray(image, current):
    """Returns a grayscale version of `image` for methods (face/contour
    detection, OCR, perceptual hashing) that need single-channel input,
    converting from whatever color space it's actually in."""
    return convert_to(image, current, 'GRAY')


def as_bgr(image, current):
    """Returns a 3-channel BGR version of `image` for methods that expect that specific layout."""
    return convert_to(image, current, 'BGR')