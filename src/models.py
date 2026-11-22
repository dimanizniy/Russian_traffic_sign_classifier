"""Model architectures."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int, weights: ResNet18_Weights | None) -> nn.Module:
    """ResNet18 with a new classification head; works best with img_size >= 96."""
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Build an untrained model; resnet18 starts from ImageNet weights."""
    if model_name == "custom_cnn":
        return CustomCNN(num_classes)
    if model_name == "residual_cnn":
        return ResidualCNN(num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes, ResNet18_Weights.IMAGENET1K_V1)
    raise ValueError(f"Unknown model: {model_name}")


def build_model_for_checkpoint(model_name: str, num_classes: int) -> nn.Module:
    """Build an architecture whose weights will be loaded from a checkpoint."""
    if model_name == "resnet18":
        return build_resnet18(num_classes, weights=None)
    return build_model(model_name, num_classes)
