"""Class-imbalance handling: per-class loss weights and per-sample sampling weights."""

from collections.abc import Sequence

import numpy as np

WEIGHTING_STRATEGIES = ("none", "inverse", "sqrt_inverse", "effective_number")


class UnknownWeightingStrategyError(ValueError):
    pass


def compute_class_weights(
    class_counts: Sequence[int],
    strategy: str,
    effective_number_beta: float = 0.999,
) -> np.ndarray:
    """Return loss weights normalised to mean 1.

    ``inverse`` (N / n_c) explodes for classes with a handful of samples, which makes
    the loss dominated by a few noisy examples. ``sqrt_inverse`` and
    ``effective_number`` (Cui et al., 2019, "Class-Balanced Loss") are softer.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every class must have at least one training sample")

    if strategy == "none":
        weights = np.ones_like(counts)
    elif strategy == "inverse":
        weights = counts.sum() / counts
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    elif strategy == "effective_number":
        effective_number = (1.0 - np.power(effective_number_beta, counts)) / (
            1.0 - effective_number_beta
        )
        weights = 1.0 / effective_number
    else:
        raise UnknownWeightingStrategyError(
            f"Unknown class weighting '{strategy}', expected one of {WEIGHTING_STRATEGIES}"
        )

    normalised: np.ndarray = weights / weights.mean()
    return normalised


def compute_sample_weights(targets: Sequence[int], class_counts: Sequence[int]) -> np.ndarray:
    """Per-sample weights giving every class equal draw probability (sqrt-damped).

    Pure 1/n_c oversampling repeats single-example classes thousands of times per
    epoch, so the square root keeps rare classes boosted without memorising them.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    per_class = 1.0 / np.sqrt(counts)
    sample_weights: np.ndarray = per_class[np.asarray(targets)]
    return sample_weights
