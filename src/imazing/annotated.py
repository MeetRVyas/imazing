"""Annotation-aware image wrapper: pairs an Imazing image with an AnnotationSet and keeps them in sync through geometry operations."""

from __future__ import annotations

import cv2
import numpy as np

from .core import Imazing
from .annotations import AnnotationSet, BoundingBox, Polygon, Keypoints, Mask


def _transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Applies a 2x3 affine or 3x3 perspective matrix to an (N, 2) array of
    points, returning the transformed (N, 2) array. Uses OpenCV.
    """
    if len(points) == 0:
        return points.reshape(-1, 2)
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    if M.shape[0] == 2:
        transformed = cv2.transform(pts, M)
    else:
        transformed = cv2.perspectiveTransform(pts, M)
    return transformed.reshape(-1, 2)


class AnnotatedImage:
    """Couples an Imazing image with its AnnotationSet, so calling
    .resize()/.crop()/.rotate()/.flip()/.pad()/.warp_perspective() on the
    pair transforms the image and updates every annotation's coordinates to
    match, instead of requiring the two to be kept in sync by hand.

    Boxes are clipped to the new frame and dropped if they end up with no area.
    Keypoints that land outside the new frame are marked not-visible (v=0) rather than dropped.
    Polygons and masks are transformed but not clipped to the frame -- masks don't need it
    a polygon whose points end up outside the frame is a known limitation:
    correctly clipping a polygon to a rectangle needs a real
    polygon-clipping algorithm (e.g. Sutherland-Hodgman), which is more
    than this needs right now.

    Any other Imazing method (blur, convert_color, draw_rect, get_stats,
    ...) is available directly through this wrapper too, delegated to the
    underlying image, since those don't move annotation coordinates.
    """

    def __init__(self, image: Imazing | str | np.ndarray | bytes, annotations: AnnotationSet | None = None):
        self.image: Imazing = image if isinstance(image, Imazing) else Imazing(image)
        self.annotations: AnnotationSet = annotations if annotations is not None else AnnotationSet()
        self._validate_masks()

    def _validate_masks(self):
        h, w = self.image.shape[:2]
        for mask in self.annotations.masks:
            if mask.data.shape[:2] != (h, w):
                raise ValueError(
                    f"Mask shape {mask.data.shape[:2]} doesn't match the image's "
                    f"current dimensions {(h, w)}."
                )

    def __repr__(self):
        h, w = self.image.shape[:2] if self.image.image is not None else (None, None)
        a = self.annotations
        return (
            f"AnnotatedImage({w}x{h}, boxes={len(a.boxes)}, polygons={len(a.polygons)}, "
            f"keypoints={len(a.keypoints)}, masks={len(a.masks)}, labels={len(a.labels)})"
        )

    def clone(self) -> "AnnotatedImage":
        return AnnotatedImage(self.image.clone(), self.annotations.copy())

    # --- Geometry ops kept in sync with annotations -------------------------

    def resize(self, width: int | None = None, height: int | None = None, inter=None) -> "AnnotatedImage":
        orig_h, orig_w = self.image.shape[:2]
        self.image.resize(width=width, height=height, inter=inter)
        new_h, new_w = self.image.shape[:2]
        sx, sy = new_w / orig_w, new_h / orig_h
        M = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float64)
        self._apply_transform(M, new_w, new_h)
        return self

    def crop(self, x: int, y: int, w: int, h: int) -> "AnnotatedImage":
        self.image.crop(x, y, w, h)
        M = np.array([[1, 0, -x], [0, 1, -y]], dtype=np.float64)
        self._apply_transform(M, w, h)
        return self

    def flip(self, horizontal: bool = True, vertical: bool = False) -> "AnnotatedImage":
        img_h, img_w = self.image.shape[:2]  # dimensions unchanged by flip
        self.image.flip(horizontal=horizontal, vertical=vertical)
        M = np.array([
            [-1 if horizontal else 1, 0, img_w if horizontal else 0],
            [0, -1 if vertical else 1, img_h if vertical else 0],
        ], dtype=np.float64)
        self._apply_transform(M, img_w, img_h)
        return self

    def rotate(self, angle: float, scale: float = 1.0) -> "AnnotatedImage":
        h, w = self.image.shape[:2]
        # Mirrors Imazing.rotate()'s own center/matrix computation exactly,
        # since that matrix isn't otherwise exposed. Keep these in sync if
        # GeometryMixin.rotate() ever changes.
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image.rotate(angle, scale=scale)
        self._apply_transform(M, w, h)
        return self

    def pad(self, top: int, bottom: int, left: int, right: int, color=(0, 0, 0)) -> "AnnotatedImage":
        self.image.pad(top, bottom, left, right, color=color)
        new_h, new_w = self.image.shape[:2]
        M = np.array([[1, 0, left], [0, 1, top]], dtype=np.float64)
        self._apply_transform(M, new_w, new_h, clip=False)
        return self

    def warp_perspective(self, src_points: np.ndarray, dst_points: np.ndarray, size: tuple[int, int]) -> "AnnotatedImage":
        # Mirrors Imazing.warp_perspective()'s own matrix computation, for
        # the same reason as rotate() above.
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.image.warp_perspective(src_points, dst_points, size)
        self._apply_transform(M, size[0], size[1])
        return self

    # --- Internal: apply a transform matrix to every annotation -------------

    def _apply_transform(self, M: np.ndarray, new_w: int, new_h: int, clip: bool = True):
        is_perspective = M.shape[0] == 3

        new_boxes: list[BoundingBox] = []
        for box in self.annotations.boxes:
            pts = _transform_points(box.corners(), M)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            new_box = BoundingBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1), label=box.label, score=box.score)
            if clip:
                new_box = new_box.clip(new_w, new_h)
            if new_box is not None:
                new_boxes.append(new_box)

        new_polygons: list[Polygon] = []
        for poly in self.annotations.polygons:
            pts = _transform_points(poly.as_array(), M)
            new_polygons.append(Polygon(points=[tuple(p) for p in pts], label=poly.label, score=poly.score))

        new_keypoints: list[Keypoints] = []
        for kp in self.annotations.keypoints:
            pts = _transform_points(kp.as_array(), M)
            new_points = []
            for (x, y), (_, _, v) in zip(pts, kp.points):
                if clip and not (0 <= x <= new_w and 0 <= y <= new_h):
                    v = 0
                new_points.append((float(x), float(y), v))
            new_keypoints.append(Keypoints(points=new_points, names=kp.names, label=kp.label))

        new_masks: list[Mask] = []
        for mask in self.annotations.masks:
            data = mask.data.astype(np.uint8)
            if is_perspective:
                warped = cv2.warpPerspective(data, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
            else:
                warped = cv2.warpAffine(data, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
            new_masks.append(Mask(data=warped.astype(mask.data.dtype), label=mask.label))

        self.annotations = AnnotationSet(
            boxes=new_boxes,
            polygons=new_polygons,
            keypoints=new_keypoints,
            masks=new_masks,
            labels=list(self.annotations.labels),
        )

    # --- Delegate everything else straight to the underlying image ----------

    def __getattr__(self, name):
        attr = getattr(self.image, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # support chaining
                # e.g. `annotated.blur().draw_rect(...)`.
                return self if result is self.image else result
            return wrapper
        return attr