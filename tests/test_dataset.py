import pytest
import torch
from PIL import Image

from dataset import dataset_metadata, get_transforms


def test_train_transform_produces_normalized_tensor():
    img = Image.new("RGB", (32, 32), color=(120, 30, 200))
    tensor = get_transforms("cifar10", train=True)(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)


def test_eval_transform_is_deterministic():
    img = Image.new("RGB", (32, 32), color=(10, 250, 5))
    t = get_transforms("cifar10", train=False)
    assert torch.equal(t(img), t(img))


def test_fashion_mnist_metadata_is_single_channel():
    meta = dataset_metadata("fashion_mnist")
    assert meta["channels"] == 1
    assert len(meta["classes"]) == 10


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_metadata("imagenet")
