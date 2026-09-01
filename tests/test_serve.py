import importlib
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from model import get_model

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "classifier_v1.pt"
    model = get_model("simplecnn", num_classes=10)
    torch.save(
        {
            "epoch": 1,
            "architecture": "simplecnn",
            "num_classes": 10,
            "dataset": "cifar10",
            "classes": CLASSES,
            "norm_mean": (0.5, 0.5, 0.5),
            "norm_std": (0.5, 0.5, 0.5),
            "in_channels": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.12,
            "val_accuracy": 0.9,
        },
        ckpt_path,
    )
    monkeypatch.setenv("CHECKPOINT_PATH", str(ckpt_path))

    import serve

    importlib.reload(serve)
    with TestClient(serve.app) as c:
        yield c


def _png_bytes(color=(128, 64, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_probability_distribution(client):
    resp = client.post("/predict", files={"image": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["probabilities"]) == set(CLASSES)
    assert body["predicted_class"] in CLASSES
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_rejects_non_image(client):
    resp = client.post("/predict", files={"image": ("x.png", b"not-an-image", "image/png")})
    assert resp.status_code == 400
