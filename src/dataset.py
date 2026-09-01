"""Dataset and DataLoader construction.

Supports CIFAR-10 (default) and Fashion-MNIST. The dataset is selected by the
``data.dataset`` key in the training config.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-dataset normalization statistics and torchvision class.
_DATASETS = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "classes": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "channels": 1,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
    # Random noise shaped like CIFAR-10. No download -- used by the Kubernetes
    # end-to-end demo and CI when the real dataset mirror is unreachable.
    "synthetic": {
        "cls": None,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "classes": [f"class_{i}" for i in range(10)],
    },
}


def _normalize_name(dataset: str) -> str:
    key = dataset.strip().lower().replace("-", "_")
    if key in ("fashionmnist", "fashion"):
        key = "fashion_mnist"
    if key in ("fake", "fakedata", "random"):
        key = "synthetic"
    if key not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_DATASETS)}")
    return key


def dataset_metadata(dataset: str) -> dict:
    """Return ``{mean, std, channels, classes}`` for a dataset name."""
    meta = _DATASETS[_normalize_name(dataset)]
    return {k: meta[k] for k in ("mean", "std", "channels", "classes")}


def get_transforms(dataset: str = "cifar10", train: bool = True) -> transforms.Compose:
    meta = _DATASETS[_normalize_name(dataset)]
    steps: list = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, pad_if_needed=True),
        ]
    elif meta["channels"] == 1:
        # keep eval spatial size consistent with the 32x32 training crop
        steps.append(transforms.Resize(32))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
    ]
    return transforms.Compose(steps)


def get_dataloaders(
    data_dir: str,
    dataset: str = "cifar10",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build the train and validation (test split) DataLoaders."""
    name = _normalize_name(dataset)
    meta = _DATASETS[name]

    if name == "synthetic":
        train_dataset = datasets.FakeData(
            1024, (3, 32, 32), 10, transform=get_transforms(dataset, train=True)
        )
        val_dataset = datasets.FakeData(
            256, (3, 32, 32), 10, transform=get_transforms(dataset, train=False)
        )
    else:
        ds_cls = meta["cls"]
        train_dataset = ds_cls(
            root=data_dir,
            train=True,
            download=download,
            transform=get_transforms(dataset, train=True),
        )
        val_dataset = ds_cls(
            root=data_dir,
            train=False,
            download=download,
            transform=get_transforms(dataset, train=False),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
