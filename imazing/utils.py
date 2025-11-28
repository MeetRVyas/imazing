import cv2
import numpy as np
import functools
from typing import Any, Callable

class ImageError(Exception):
    """Custom exception for Imazing library errors."""
    pass

def validate_image(image: np.ndarray) -> None:
    """
    Strict validation of a numpy image array.
    Raises ImageError if invalid.
    """
    if image is None:
        raise ImageError("Image is None. The previous operation may have failed.")

    if not isinstance(image, np.ndarray):
        raise ImageError(f"Invalid image type: {type(image)}. Expected numpy.ndarray.")

    if image.size == 0:
        raise ImageError("Image is empty (0 pixels).")

    if len(image.shape) < 2:
        raise ImageError(f"Invalid image shape: {image.shape}. Expected at least 2 dimensions.")

def ensure_valid(func: Callable) -> Callable:
    """
    Decorator to ensure the image is valid BEFORE and AFTER an operation.
    Use this on any method that modifies self.image.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1. Pre-Check
        try:
            validate_image(self.image)
        except ImageError as e:
            raise ImageError(f"Cannot perform '{func.__name__}': {str(e)}")

        # 2. Execute
        result = func(self, *args, **kwargs)

        # 3. Post-Check (Only if the method returns 'self' or modifies state)
        # We assume if it returns self, it modified the internal image.
        if result is self:
            try:
                validate_image(self.image)
            except ImageError as e:
                raise ImageError(f"Operation '{func.__name__}' resulted in an invalid image: {str(e)}")

        return result
    return wrapper

def check_dependency(package_name: str):
    """
    Helper to check if an optional dependency exists.
    Returns the module if found, else None.
    """
    import importlib
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None
