"""Annotation primitives: bounding boxes, polygons, keypoints, masks, and whole-image labels,
plus a per-image container (AnnotationSet) that groups them together."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
    """An axis-aligned box in absolute pixel coordinates.
    x -> x + width
    y -> y + height"""

    x: float
    y: float
    width: float
    height: float
    label: str | None = None
    score: float | None = None

    def __post_init__(self):
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"BoundingBox width/height must be non-negative, got width={self.width}, height={self.height}."
            )

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def corners(self) -> np.ndarray:
        """The 4 corners as an (4, 2) array: TL, TR, BR, BL."""
        return np.array([
            [self.x, self.y],
            [self.x2, self.y],
            [self.x2, self.y2],
            [self.x, self.y2],
        ], dtype=np.float64)

    def clip(self, width: float, height: float) -> "BoundingBox" | None:
        """Clips this box to a `width` x `height` frame.

        Returns None if the clipped box has no area (i.e. it's fully
        outside the frame), so callers can drop it from a result list.
        """
        x1, y1 = max(self.x, 0.0), max(self.y, 0.0)
        x2, y2 = min(self.x2, width), min(self.y2, height)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2 - x1, y2 - y1, label=self.label, score=self.score)


@dataclass
class Polygon:
    """A closed polygon as a sequence of (x, y) pixel points."""

    points: list[tuple[float, float]]
    label: str | None = None
    score: float | None = None

    def as_array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64).reshape(-1, 2)

    def bounding_box(self) -> BoundingBox:
        """The smallest axis-aligned box enclosing this polygon."""
        arr = self.as_array()
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return BoundingBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1), label=self.label)


@dataclass
class Keypoints:
    """A named or unnamed set of (x, y, visible) points -- e.g. a pose
    skeleton. `visible` follows the common COCO-style convention where 0
    means not visible / not labeled, and any nonzero value means visible.
    """

    points: list[tuple[float, float, int]]
    names: list[str] | None = None
    label: str | None = None

    def as_array(self) -> np.ndarray:
        return np.asarray([(x, y) for x, y, _ in self.points], dtype=np.float64).reshape(-1, 2)


@dataclass
class Mask:
    """A binary segmentation mask, stored as a full-image-sized array so it composes
    directly with the same geometry ops that already work on `Imazing.image`."""

    data: np.ndarray  # 2D array, same H x W as the image it annotates
    label: str | None = None

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError(f"Mask.data must be a 2D array, got shape {self.data.shape}.")


@dataclass
class Label:
    """An image-level classification label with no associated geometry
    (e.g. 'cat' for a whole photo, as opposed to a labeled region)."""

    name: str
    score: float | None = None


Annotation = BoundingBox | Polygon | Keypoints | Mask | Label


@dataclass
class AnnotationSet:
    """All annotations belonging to a single image."""

    boxes: list[BoundingBox] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    keypoints: list[Keypoints] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)

    def add(self, annotation: Annotation) -> "AnnotationSet":
        """Appends an annotation to the list matching its type."""
        if isinstance(annotation, BoundingBox):
            self.boxes.append(annotation)
        elif isinstance(annotation, Polygon):
            self.polygons.append(annotation)
        elif isinstance(annotation, Keypoints):
            self.keypoints.append(annotation)
        elif isinstance(annotation, Mask):
            self.masks.append(annotation)
        elif isinstance(annotation, Label):
            self.labels.append(annotation)
        else:
            raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")
        return self

    def is_empty(self) -> bool:
        return not (self.boxes or self.polygons or self.keypoints or self.masks or self.labels)

    def copy(self) -> "AnnotationSet":
        """An independent deep copy."""
        return copy.deepcopy(self)