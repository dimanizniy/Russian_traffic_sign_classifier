import numpy as np
import pytest

from evaluate import summarize_metrics


def test_macro_f1_ignores_classes_absent_from_val() -> None:
    targets = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 0])

    summary, report = summarize_metrics(targets, preds, num_classes=4)

    assert summary["val_classes_present"] == 2
    assert summary["val_accuracy"] == pytest.approx(0.75)
    assert summary["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert summary["macro_f1_all_train_classes"] == pytest.approx((0.8 + 2 / 3) / 4)
    assert set(report) >= {"0", "1", "macro avg"}
