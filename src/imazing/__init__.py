import logging

from .core import Imazing
from .video import VideoStream
from .exceptions import ImazingError, ImageLoadError, NoImageLoadedError

__version__ = "1.0.0"

__all__ = [
    'Imazing',
    'VideoStream',
    'ImazingError',
    'ImageLoadError',
    'NoImageLoadedError',
]


logging.getLogger(__name__).addHandler(logging.NullHandler())