"""Aggregate statistics over a Dataset: dimension distributions, per-channel
pixel statistics, label/class balance, and a corrupt-item count."""

from __future__ import annotations

import random as _random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetStats:
    num_items: int
    num_sampled: int
    num_corrupt: int
    width_min: int | None
    width_max: int | None
    width_mean: float | None
    height_min: int | None
    height_max: int | None
    height_mean: float | None
    color_spaces: Counter = field(default_factory=Counter)
    channel_counts: Counter = field(default_factory=Counter)
    label_counts: Counter = field(default_factory=Counter)
    box_count_total: int = 0
    per_channel_mean: list[float] | None = None
    per_channel_std: list[float] | None = None


def compute_dataset_stats(dataset, sample: int | None = None, seed: int | None = None) -> DatasetStats:
    """Computes DatasetStats over `dataset`."""
    n = len(dataset)
    indices = list(range(n))
    if sample is not None and sample < n:
        indices = _random.Random(seed).sample(indices, sample)

    widths: list[int] = []
    heights: list[int] = []
    color_spaces: Counter = Counter()
    channel_counts: Counter = Counter()
    label_counts: Counter = Counter()
    box_count_total = 0
    num_corrupt = 0
    channel_sums: dict[int, np.ndarray] = {}
    channel_sq_sums: dict[int, np.ndarray] = {}
    channel_pixel_counts: dict[int, int] = {}

    for i in indices:
        try:
            item = dataset[i]
            img = item.image
            if img.image is None:
                raise ValueError("loaded image is empty")
        except Exception:
            num_corrupt += 1
            continue

        h, w = img.image.shape[:2]
        widths.append(w)
        heights.append(h)
        color_spaces[img.color_space] += 1
        channel_counts[img.channels] += 1

        for label in item.annotations.labels:
            label_counts[label.name] += 1
        for box in item.annotations.boxes:
            box_count_total += 1
            if box.label is not None:
                label_counts[box.label] += 1
        for poly in item.annotations.polygons:
            if poly.label is not None:
                label_counts[poly.label] += 1
        for mask in item.annotations.masks:
            if mask.label is not None:
                label_counts[mask.label] += 1

        c = img.channels
        flat = img.image.reshape(-1, c).astype(np.float64)
        if c not in channel_sums:
            channel_sums[c] = np.zeros(c)
            channel_sq_sums[c] = np.zeros(c)
            channel_pixel_counts[c] = 0
        channel_sums[c] += flat.sum(axis=0)
        channel_sq_sums[c] += (flat ** 2).sum(axis=0)
        channel_pixel_counts[c] += flat.shape[0]

    per_channel_mean = None
    per_channel_std = None
    if channel_sums:
        # Use whichever channel count is most common so a mixed-channel
        # dataset still gets a sensible headline number;
        # the per-channel-count breakdown remains available via `channel_counts`.
        dominant_c = channel_counts.most_common(1)[0][0]
        total_pixels = channel_pixel_counts[dominant_c]
        mean = channel_sums[dominant_c] / total_pixels
        var = channel_sq_sums[dominant_c] / total_pixels - mean ** 2
        per_channel_mean = mean.tolist()
        per_channel_std = np.sqrt(np.maximum(var, 0)).tolist()

    return DatasetStats(
        num_items=n,
        num_sampled=len(indices),
        num_corrupt=num_corrupt,
        width_min=min(widths) if widths else None,
        width_max=max(widths) if widths else None,
        width_mean=float(np.mean(widths)) if widths else None,
        height_min=min(heights) if heights else None,
        height_max=max(heights) if heights else None,
        height_mean=float(np.mean(heights)) if heights else None,
        color_spaces=color_spaces,
        channel_counts=channel_counts,
        label_counts=label_counts,
        box_count_total=box_count_total,
        per_channel_mean=per_channel_mean,
        per_channel_std=per_channel_std,
    )