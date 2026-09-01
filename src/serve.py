"""FastAPI model server.

Loads a checkpoint saved by ``train.py`` and exposes:
    GET  /health   -> 200 when the model is loaded, 503 otherwise
    POST /predict   -> multipart form field ``image``; returns class probabilities

Checkpoint path resolution (first match wins):
    1. $CHECKPOINT_PATH
    2. {output.checkpoint_dir}/{output.model_name} from the training config
    3. /app/checkpoints/classifier_v1.pt
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

_CONFIG_CANDIDATES = (
    os.environ.get("TRAINING_CONFIG_PATH"),
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
)


def _resolve_checkpoint_path() -> Path:
    if os.environ.get("CHECKPOINT_PATH"):
        return Path(os.environ["CHECKPOINT_PATH"])
    for candidate in _CONFIG_CANDIDATES:
        if candidate and Path(candidate).is_file():
            cfg = yaml.safe_load(Path(candidate).read_text())
            out = cfg["output"]
            return Path(out["checkpoint_dir"]) / out["model_name"]
    return Path("/app/checkpoints/classifier_v1.pt")


class Classifier:
    """Wraps a loaded checkpoint and turns an image into class probabilities."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.classes: list[str] = ckpt.get("classes") or [
            str(i) for i in range(ckpt["num_classes"])
        ]
        self.in_channels: int = int(ckpt.get("in_channels", 3))

        self.model = get_model(
            architecture=ckpt["architecture"],
            num_classes=ckpt["num_classes"],
            in_channels=self.in_channels,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        mean = ckpt.get("norm_mean", (0.4914, 0.4822, 0.4465))
        std = ckpt.get("norm_std", (0.2470, 0.2435, 0.2616))
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.metadata = {
            "architecture": ckpt["architecture"],
            "dataset": ckpt.get("dataset"),
            "num_classes": ckpt["num_classes"],
            "trained_epochs": ckpt.get("epoch"),
            "val_accuracy": ckpt.get("val_accuracy"),
        }

    @torch.no_grad()
    def predict(self, raw: bytes) -> dict:
        mode = "L" if self.in_channels == 1 else "RGB"
        try:
            image = Image.open(io.BytesIO(raw)).convert(mode)
        except Exception as exc:  # noqa: BLE001 - any decode failure is a bad request
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1).squeeze(0).cpu().tolist()
        paired = list(zip(self.classes, probs, strict=True))
        ranked = sorted(paired, key=lambda kv: kv[1], reverse=True)
        return {
            "predicted_class": ranked[0][0],
            "confidence": round(ranked[0][1], 6),
            "probabilities": {name: round(p, 6) for name, p in paired},
            "top_k": [{"class": n, "probability": round(p, 6)} for n, p in ranked[:5]],
        }


_classifier: Classifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    path = _resolve_checkpoint_path()
    if path.is_file():
        _classifier = Classifier(path)
        print(f"[serve] loaded checkpoint {path}", flush=True)
    else:
        # Stay up with no model: the readiness probe fails on /health -> 503, so
        # k8s keeps the pod out of the Service until a checkpoint appears / restart.
        print(f"[serve] checkpoint not found at {path}; model unavailable", flush=True)
    yield


app = FastAPI(title="mlops-pytorch-pipeline serving", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if _classifier is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.get("/")
def info():
    if _classifier is None:
        return {"model_loaded": False}
    return {"model_loaded": True, **_classifier.metadata}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _classifier.predict(await image.read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",  # noqa: S104 - containerized service, bind all interfaces
        port=int(os.environ.get("PORT", "8080")),
    )
