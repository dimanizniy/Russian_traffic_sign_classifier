"""Loss functions for imbalanced classification."""

import torch
import torch.nn.functional as functional
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights."""

    weight: torch.Tensor | None

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample_ce = functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        target_probability = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        focal = (1.0 - target_probability).pow(self.gamma) * per_sample_ce

        if self.weight is None:
            return focal.mean()
        sample_weight = self.weight[targets]
        return (focal * sample_weight).sum() / sample_weight.sum()


def build_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss: {loss_name}")
