"""Shared fixtures for the imazing test suite."""

import numpy as np
import pytest


@pytest.fixture
def color_image() -> np.ndarray:
    """A small deterministic BGR image with some real structure (not pure
    noise), so operations like face/contour detection have something to
    chew on rather than immediately returning nothing."""
    rng = np.random.default_rng(seed=42)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    img[30:90, 40:100] = (255, 255, 255)  # a bright rectangle
    return img


@pytest.fixture
def gray_image() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return (rng.random((120, 160)) * 255).astype(np.uint8)


@pytest.fixture
def small_color_image() -> np.ndarray:
    rng = np.random.default_rng(seed=1)
    return (rng.random((20, 20, 3)) * 255).astype(np.uint8)
