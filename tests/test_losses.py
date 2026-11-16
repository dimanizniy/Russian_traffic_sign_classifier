import pytest
import torch
from torch import nn

from losses import FocalLoss, build_criterion


def test_focal_with_zero_gamma_equals_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    weight = torch.rand(5) + 0.5

    expected = nn.CrossEntropyLoss(weight=weight)(logits, targets)
    actual = FocalLoss(gamma=0.0, weight=weight)(logits, targets)

    assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


def test_focal_downweights_confident_correct_predictions() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.5, 0.0]])
    targets = torch.tensor([0, 0])
    ce = nn.CrossEntropyLoss()(logits, targets)
    focal = FocalLoss(gamma=2.0)(logits, targets)
    assert focal.item() < ce.item()


def test_build_criterion_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError):
        build_criterion("hinge", None, 0.0, 2.0)
