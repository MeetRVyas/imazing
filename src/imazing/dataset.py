"""A lazy, indexable collection of (image, annotations) pairs.

Dataset stores lightweight references -- a file path, or an already-loaded
image source, plus an AnnotationSet -- rather than decoded pixel data.
Nothing is actually loaded into memory until an item is indexed, so
constructing a Dataset over a large directory of images is cheap; the cost
shows up per-item, at __getitem__ time.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from .annotated import AnnotatedImage
from .annotations import AnnotationSet, Label
from .exceptions import ImazingError

DEFAULT_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

ItemSource = str | np.ndarray | bytes


@dataclass
class DatasetValidationReport:
    """The result of Dataset.validate(): which items failed to load and why."""

    total: int
    errors: list[tuple[int, ItemSource, str]] = field(default_factory=list)

    @property
    def num_errors(self) -> int:
        return len(self.errors)

    @property
    def is_clean(self) -> bool:
        return self.num_errors == 0


class Dataset:
    """A lazily-loaded sequence of AnnotatedImage items."""

    def __init__(self, items: list[tuple[ItemSource, AnnotationSet]] | None = None):
        self._items: list[tuple[ItemSource, AnnotationSet]] = list(items) if items else []

    # --- Construction --------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        sources: Iterable[ItemSource],
        annotations: Iterable[AnnotationSet] | None = None,
    ) -> "Dataset":
        """Builds a Dataset from a plain list of image sources (paths,
        arrays, or bytes), optionally paired with an AnnotationSet each."""
        sources = list(sources)
        if annotations is None:
            annotation_list = [AnnotationSet() for _ in sources]
        else:
            annotation_list = list(annotations)
            if len(annotation_list) != len(sources):
                raise ValueError(
                    f"Got {len(sources)} sources but {len(annotation_list)} annotation sets; "
                    "they must be the same length."
                )
        return cls(list(zip(sources, annotation_list)))

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        pattern: str | None = None,
        recursive: bool = False,
        extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> "Dataset":
        """Builds a Dataset (no annotations) from every image file in
        `directory`. Matches `pattern` (a glob) if given, otherwise every
        file whose suffix is in `extensions`."""
        directory = Path(directory)
        glob_fn = directory.rglob if recursive else directory.glob
        if pattern is not None:
            paths = sorted(p for p in glob_fn(pattern) if p.is_file())
        else:
            exts = {e.lower() for e in extensions}
            paths = sorted(p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in exts)
        return cls.from_paths([str(p) for p in paths])

    @classmethod
    def from_class_folders(
        cls,
        root: str | Path,
        extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> "Dataset":
        """Builds a classification Dataset from an ImageFolder-style
        layout: `root/<class_name>/<image files>`. Each item gets a single
        `Label(name=<class_name>)` annotation."""
        root = Path(root)
        exts = {e.lower() for e in extensions}
        items = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in exts:
                    items.append((str(image_path), AnnotationSet(labels=[Label(name=class_dir.name)])))
        return cls(items)

    # --- Sequence protocol -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return self.subset(range(*index.indices(len(self))))
        source, annotations = self._items[index]
        return AnnotatedImage(source, annotations.copy())

    def __iter__(self) -> Iterator[AnnotatedImage]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"Dataset({len(self)} items)"

    def sources(self) -> list[ItemSource]:
        """The raw (unloaded) sources backing this dataset, in order."""
        return [source for source, _ in self._items]

    # --- Batching / splitting / subsetting --------------------------------------

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        """Yields lists of AnnotatedImage of length `batch_size` (the last
        batch may be shorter, unless drop_last=True)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        order = list(range(len(self)))
        if shuffle:
            _random.Random(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            yield [self[i] for i in chunk]

    def subset(self, indices: Iterable[int]) -> "Dataset":
        """A new Dataset containing only the given indices, in that order."""
        return Dataset([self._items[i] for i in indices])

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int | None = None,
    ) -> tuple["Dataset", "Dataset", "Dataset"]:
        """Splits into (train, val, test) Datasets by shuffled fraction.
        train + val + test must sum to 1.0 (within floating-point
        tolerance)."""
        total = train + val + test
        if not np.isclose(total, 1.0):
            raise ValueError(f"train + val + test must sum to 1.0, got {total}.")
        order = list(range(len(self)))
        _random.Random(seed).shuffle(order)
        n_train = round(train * len(order))
        n_val = round(val * len(order))
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        return self.subset(train_idx), self.subset(val_idx), self.subset(test_idx)

    def filter(self, predicate: Callable[[AnnotatedImage], bool]) -> "Dataset":
        """A new Dataset containing only items for which `predicate`
        returns True. `predicate` receives a loaded AnnotatedImage, so it
        can inspect pixel data as well as annotations (at the cost of
        decoding every item to check)."""
        keep = [i for i in range(len(self)) if predicate(self[i])]
        return self.subset(keep)

    # --- Cleaning ----------------------------------------------------------------

    def validate(self) -> DatasetValidationReport:
        """Attempts to load every item and reports which ones fail, and why."""
        errors = []
        for i, (source, _) in enumerate(self._items):
            try:
                item = self[i]
                if item.image.image is None:
                    raise ImazingError("loaded image is empty")
            except Exception as exc:
                errors.append((i, source, str(exc)))
        return DatasetValidationReport(total=len(self), errors=errors)

    def drop_invalid(self) -> "Dataset":
        """A new Dataset excluding any item that fails to load."""
        report = self.validate()
        bad = {i for i, _, _ in report.errors}
        keep = [i for i in range(len(self)) if i not in bad]
        return self.subset(keep)

    def deduplicate(self, threshold: int = 0) -> "Dataset":
        """A new Dataset with duplicate images removed, using the
        perceptual hash from Imazing.compute_hash(). `threshold=0` (the
        default) only removes exact hash matches (O(n), via a set).
        `threshold > 0` also removes any hash within that Hamming distance
        of a hash already kept -- O(n^2), fine for small/medium datasets,
        not built to scale to huge ones."""
        kept_hashes: list[int] = []
        keep_indices: list[int] = []
        for i in range(len(self)):
            h = self[i].image.compute_hash()
            if threshold == 0:
                is_duplicate = h in kept_hashes
            else:
                is_duplicate = any(bin(h ^ kh).count("1") <= threshold for kh in kept_hashes)
            if not is_duplicate:
                kept_hashes.append(h)
                keep_indices.append(i)
        return self.subset(keep_indices)

    # --- Statistics ----------------------------------------------------------------

    def stats(self, sample: int | None = None, seed: int | None = None):
        """Computes aggregate statistics over this dataset. See
        imazing.dataset_stats.compute_dataset_stats for details."""
        from .dataset_stats import compute_dataset_stats
        return compute_dataset_stats(self, sample=sample, seed=seed)

    # --- Import / export (format codecs live in dataset_formats.py) ----------------

    @classmethod
    def from_coco(cls, images_dir: str | Path, annotation_file: str | Path) -> "Dataset":
        from .dataset_formats import load_coco
        return load_coco(images_dir, annotation_file)

    def to_coco(self, output_dir: str | Path, annotation_file: str = "annotations.json", copy_images: bool = True):
        from .dataset_formats import save_coco
        return save_coco(self, output_dir, annotation_file, copy_images=copy_images)

    @classmethod
    def from_yolo(cls, images_dir: str | Path, labels_dir: str | Path, classes_file: str | Path) -> "Dataset":
        from .dataset_formats import load_yolo
        return load_yolo(images_dir, labels_dir, classes_file)

    def to_yolo(self, output_dir: str | Path, class_names: Sequence[str], copy_images: bool = True):
        from .dataset_formats import save_yolo
        return save_yolo(self, output_dir, class_names, copy_images=copy_images)

    @classmethod
    def from_voc(cls, images_dir: str | Path, annotations_dir: str | Path) -> "Dataset":
        from .dataset_formats import load_voc
        return load_voc(images_dir, annotations_dir)

    def to_voc(self, output_dir: str | Path, copy_images: bool = True):
        from .dataset_formats import save_voc
        return save_voc(self, output_dir, copy_images=copy_images)

    @classmethod
    def from_image_mask_folders(cls, images_dir: str | Path, masks_dir: str | Path) -> "Dataset":
        from .dataset_formats import load_image_mask_folders
        return load_image_mask_folders(images_dir, masks_dir)

    def to_image_mask_folders(self, output_dir: str | Path):
        from .dataset_formats import save_image_mask_folders
        return save_image_mask_folders(self, output_dir)

    def to_class_folders(self, output_dir: str | Path):
        from .dataset_formats import save_class_folders
        return save_class_folders(self, output_dir)