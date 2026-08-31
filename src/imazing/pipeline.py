"""A small, reproducible transform/pipeline API layered on top of existing methods.
Just a way to sequence, randomize, and reproduce calls to methods that already exist."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


# --- Random value samplers ---------------------------------------------------
#
# Any object exposing `.sample(rng) -> value` can be used as a kwarg value in a Transform.
# It's resolved to a concrete value through the pipeline's RNG each time the transform runs.
# (a continuous range, an integer range, a fixed set of choices);
# A custom sampler is just another object with a `.sample(rng)` method.

class Sampler(Protocol):
    def sample(self, rng: np.random.Generator) -> Any:
        ...

class Uniform(Sampler):
    """Samples a float uniformly from [low, high) each time it's used."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class RandInt(Sampler):
    """Samples an int uniformly from [low, high] (inclusive) each time."""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))

    def __repr__(self):
        return f"RandInt({self.low}, {self.high})"


class Choice(Sampler):
    """Samples one value from a fixed list of options each time."""

    def __init__(self, options: Sequence[Any]):
        if len(options) == 0:
            raise ValueError("Choice requires at least one option.")
        self.options = list(options)

    def sample(self, rng: np.random.Generator) -> Any:
        idx = int(rng.integers(0, len(self.options)))
        return self.options[idx]

    def __repr__(self):
        return f"Choice({self.options})"


# --- Shared seed/RNG handling -----------------------------------------------

class _SeedableStep:
    """Shared seed/RNG handling for pipeline steps that can be used either
    standalone (owning their own seeded RNG) or nested inside a parent
    Compose/OneOf (using whatever RNG the parent passes down)."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Re-seeds this step's own RNG, e.g. to reproduce an earlier run
        exactly. Only affects calls where this step supplies its own RNG
        (called directly) -- a nested step uses its parent's RNG instead."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _resolve_rng(self, rng: np.random.Generator | None) -> np.random.Generator:
        return rng if rng is not None else self._rng


def _validate_probability(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0.0 and 1.0, got {p}.")


# --- Pipeline steps ----------------------------------------------------------

class Transform(_SeedableStep):
    """A single named operation, applied with probability `p`."""

    def __init__(self, name: str, p: float = 1.0, seed: int | None = None, **kwargs):
        super().__init__(seed=seed)
        _validate_probability(p)
        self.name = name
        self.p = p
        self.kwargs = kwargs

    def __call__(self, image, rng: np.random.Generator | None = None):
        rng = self._resolve_rng(rng)
        if rng.random() >= self.p:
            return image
        resolved_kwargs = {
            key: (value.sample(rng) if hasattr(value, "sample") else value)
            for key, value in self.kwargs.items()
        }
        method = getattr(image, self.name)
        return method(**resolved_kwargs)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"Transform({self.name!r}, p={self.p}, {args})"


class OneOf(_SeedableStep):
    """Selects exactly one of several steps at random (optionally weighted) and applies it,
    itself gated by an overall probability `p`.

    Each entry in `transforms` can be a `Transform`, another `Compose`, or
    another `OneOf` -- anything with a `(image, rng=None)` call signature.
    """

    def __init__(
        self,
        transforms: Sequence[Any],
        weights: Sequence[float] | None = None,
        p: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        _validate_probability(p)
        self.transforms = list(transforms)
        if weights is not None and len(weights) != len(self.transforms):
            raise ValueError(
                f"weights (length {len(weights)}) must match transforms (length {len(self.transforms)})."
            )
        self.weights = list(weights) if weights is not None else None
        self.p = p

    def __call__(self, image, rng: np.random.Generator | None = None):
        rng = self._resolve_rng(rng)
        if rng.random() >= self.p:
            return image
        if not self.transforms:
            return image

        if self.weights is not None:
            probs = np.asarray(self.weights, dtype=np.float64)
            probs = probs / probs.sum()
            idx = int(rng.choice(len(self.transforms), p=probs))
        else:
            idx = int(rng.integers(0, len(self.transforms)))

        return self.transforms[idx](image, rng)

    def __repr__(self):
        return f"OneOf({self.transforms!r}, weights={self.weights}, p={self.p})"


class Compose(_SeedableStep):
    """Applies a sequence of steps in order.

    Can be used standalone (`pipeline(image)`, using its own seeded RNG)
    or nested inside another Compose/OneOf, in which case it uses whatever
    RNG its parent passes down.
    """

    def __init__(self, transforms: Sequence[Any], seed: int | None = None):
        super().__init__(seed=seed)
        self.transforms = list(transforms)

    def __call__(self, image, rng: np.random.Generator | None = None):
        rng = self._resolve_rng(rng)
        for step in self.transforms:
            image = step(image, rng)
        return image

    def __repr__(self):
        steps = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{steps}])"