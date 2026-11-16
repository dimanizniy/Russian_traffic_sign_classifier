import numpy as np
import pytest

from class_balance import (
    UnknownWeightingStrategyError,
    compute_class_weights,
    compute_sample_weights,
)

COUNTS = [1000, 100, 10, 1]


@pytest.mark.parametrize("strategy", ["none", "inverse", "sqrt_inverse", "effective_number"])
def test_weights_have_unit_mean(strategy: str) -> None:
    weights = compute_class_weights(COUNTS, strategy)
    assert weights.mean() == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", ["inverse", "sqrt_inverse", "effective_number"])
def test_rarer_classes_get_larger_weights(strategy: str) -> None:
    weights = compute_class_weights(COUNTS, strategy)
    assert np.all(np.diff(weights) > 0)


def test_softer_strategies_reduce_weight_spread() -> None:
    def spread(strategy: str) -> float:
        weights = compute_class_weights(COUNTS, strategy)
        return float(weights.max() / weights.min())

    assert spread("inverse") == pytest.approx(1000)
    assert spread("sqrt_inverse") < spread("inverse")
    assert spread("effective_number") < spread("inverse")


def test_unknown_strategy_raises() -> None:
    with pytest.raises(UnknownWeightingStrategyError):
        compute_class_weights(COUNTS, "magic")


def test_empty_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_class_weights([5, 0], "inverse")


def test_sample_weights_follow_sqrt_inverse_frequency() -> None:
    weights = compute_sample_weights([0, 0, 0, 0, 1], class_counts=[4, 1])
    assert weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0])
