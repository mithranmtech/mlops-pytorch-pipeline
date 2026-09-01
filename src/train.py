"""Training entry point.

Reads hyperparameters from a YAML config, trains an image classifier, logs one
JSON object per epoch to stdout (JSON Lines), and saves the best checkpoint.

Config path resolution (first match wins):
    1. $TRAINING_CONFIG_PATH
    2. /app/configs/training_config.yaml   (k8s ConfigMap / Docker mount)
    3. configs/training_config.yaml        (local run from repo root)

Set FAST_DEV_RUN=1 to run a single short epoch on a handful of batches -- used
by CI and the Docker build verification so they finish in seconds.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import dataset_metadata, get_dataloaders
from model import get_model

_CONFIG_CANDIDATES = (
    os.environ.get("TRAINING_CONFIG_PATH"),
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
)


def log(**fields) -> None:
    """Emit a single JSON line to stdout."""
    print(json.dumps(fields), flush=True)


def resolve_config_path() -> Path:
    for candidate in _CONFIG_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"no training config found; looked at {[c for c in _CONFIG_CANDIDATES if c]}"
    )


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main() -> None:
    fast_dev_run = os.environ.get("FAST_DEV_RUN", "").lower() in ("1", "true", "yes")

    config_path = resolve_config_path()
    config = load_config(config_path)
    log(event="config_loaded", path=str(config_path), fast_dev_run=fast_dev_run)

    m_cfg, t_cfg = config["model"], config["training"]
    d_cfg, o_cfg = config["data"], config["output"]

    seed = int(t_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = dataset_metadata(d_cfg["dataset"])

    model = get_model(
        architecture=m_cfg["architecture"],
        num_classes=m_cfg["num_classes"],
        in_channels=meta["channels"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=d_cfg["data_dir"],
        dataset=d_cfg["dataset"],
        batch_size=t_cfg["batch_size"],
        num_workers=int(d_cfg.get("num_workers", 2)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=float(t_cfg.get("weight_decay", 5e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    writer = None
    tb_dir = o_cfg.get("tensorboard_dir")
    if tb_dir:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        writer = SummaryWriter(tb_dir)

    epochs = 1 if fast_dev_run else int(t_cfg["epochs"])
    max_batches = 5 if fast_dev_run else None
    patience = int(t_cfg["early_stopping_patience"])

    checkpoint_dir = Path(o_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / o_cfg["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    log(event="training_start", device=str(device), epochs=epochs, params=sum(
        p.numel() for p in model.parameters()))

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, max_batches
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, None, max_batches
        )
        log(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_acc, 4),
            seconds=round(time.time() - started, 1),
        )
        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": m_cfg["architecture"],
                    "num_classes": m_cfg["num_classes"],
                    "dataset": d_cfg["dataset"],
                    "classes": meta["classes"],
                    "norm_mean": meta["mean"],
                    "norm_std": meta["std"],
                    "in_channels": meta["channels"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            log(event="checkpoint_saved", path=str(checkpoint_path), val_loss=round(val_loss, 4))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch)
                break

    if writer:
        writer.close()

    if not checkpoint_path.exists():
        log(event="error", message="training finished without saving a checkpoint")
        sys.exit(1)

    log(event="training_complete", best_val_loss=round(best_val_loss, 4),
        checkpoint=str(checkpoint_path))


if __name__ == "__main__":
    main()
