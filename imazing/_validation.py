"""Shared helpers. Not part of the public API."""

import functools

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