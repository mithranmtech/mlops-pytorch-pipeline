import pytest
import torch
import torch.nn as nn

from model import SimpleCNN, get_model


@pytest.mark.parametrize("architecture", ["resnet18", "ResNet18", "simplecnn"])
def test_forward_pass_output_shape(architecture):
    model = get_model(architecture, num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_resnet18_stem_is_adapted_for_cifar():
    """Stock ResNet-18 uses a 7x7 stride-2 stem + maxpool; ours must be 3x3 / no pool."""
    model = get_model("resnet18", num_classes=10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_num_classes_is_respected():
    model = get_model("simplecnn", num_classes=7)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 7)


def test_single_channel_input():
    model = get_model("resnet18", num_classes=10, in_channels=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 10)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="unknown architecture"):
        get_model("transformer-xl", num_classes=10)


def test_simplecnn_is_lightweight():
    params = sum(p.numel() for p in SimpleCNN(10).parameters())
    assert params < 1_000_000
