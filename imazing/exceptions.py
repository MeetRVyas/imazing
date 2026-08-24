"""Custom exception hierarchy for imazing."""


class ImazingError(Exception):
    """Base class for all imazing-specific errors."""


class ImageLoadError(ImazingError, ValueError):
    """Raised when an image source cannot be loaded or decoded.

    Subclasses ValueError for backward compatibility with the previous
    behavior of ``Imazing.load()``.
    """


class NoImageLoadedError(ImazingError, RuntimeError):
    """Raised when an operation is attempted before an image is loaded."""