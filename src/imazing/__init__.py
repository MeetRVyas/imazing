import logging

from .core import Imazing
from .video import VideoStream
from .exceptions import ImazingError, ImageLoadError, NoImageLoadedError
from .annotations import AnnotationSet, BoundingBox, Polygon, Keypoints, Mask, Label
from .annotated import AnnotatedImage
from .pipeline import Transform, Compose, OneOf, Uniform, RandInt, Choice
from .dataset import Dataset, DatasetValidationReport
from .dataset_stats import DatasetStats

__version__ = "1.0.0"

__all__ = [
    'Imazing',
    'VideoStream',
    'ImazingError',
    'ImageLoadError',
    'NoImageLoadedError',
    'AnnotatedImage',
    'AnnotationSet',
    'BoundingBox',
    'Polygon',
    'Keypoints',
    'Mask',
    'Label',
    'Transform',
    'Compose',
    'OneOf',
    'Uniform',
    'RandInt',
    'Choice',
    'Dataset',
    'DatasetValidationReport',
    'DatasetStats',
]


logging.getLogger(__name__).addHandler(logging.NullHandler())