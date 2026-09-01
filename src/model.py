"""Model definitions for the image classifier.

`get_model` is the single entry point used by both training (`train.py`) and
serving (`serve.py`). It maps an ``architecture`` string from the config file to
a ready-to-use ``nn.Module``.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """Small VGG-style CNN, a lightweight alternative to ResNet-18 for 32x32 images."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # noqa: D102
        return self.classifier(self.features(x))


def _cifar_resnet18(num_classes: int, in_channels: int = 3) -> nn.Module:
    """torchvision ResNet-18 adapted for small (32x32) inputs.

    The stock ResNet-18 stem (7x7 stride-2 conv + 3x3 stride-2 maxpool) throws
    away most of a 32x32 image before the first residual block. Swapping in a
    3x3 stride-1 conv and dropping the maxpool is the standard "CIFAR ResNet"
    fix and recovers most of the accuracy gap.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


_BUILDERS = {
    "resnet18": _cifar_resnet18,
    "simplecnn": lambda num_classes, in_channels=3: SimpleCNN(num_classes, in_channels),
    "cnn": lambda num_classes, in_channels=3: SimpleCNN(num_classes, in_channels),
}


def get_model(architecture: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    """Build a model by name.

    Args:
        architecture: one of ``resnet18``, ``simplecnn`` (case-insensitive).
        num_classes: number of output classes.
        in_channels: input image channels (3 for CIFAR-10, 1 for Fashion-MNIST).
    """
    key = architecture.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected one of {sorted(_BUILDERS)}"
        )
    return _BUILDERS[key](num_classes, in_channels)
