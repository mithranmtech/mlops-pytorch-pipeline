#!/usr/bin/env bash
# Local end-to-end check: build both images, run a fast training pass, serve the
# checkpoint, hit /health and /predict. Mirrors the assignment "Verification" block.
set -euo pipefail
cd "$(dirname "$0")/.."

IMG_TRAIN=${IMG_TRAIN:-mlops-train:v1}
IMG_SERVE=${IMG_SERVE:-mlops-serve:v1}
# SKIP_BUILD=1 / SKIP_TRAIN=1 let you re-run just the serving check.

mkdir -p data checkpoints

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "==> build training image"
  docker build -f docker/Dockerfile.train -t "$IMG_TRAIN" .
  echo "==> build serving image"
  docker build -f docker/Dockerfile.serve -t "$IMG_SERVE" .
fi

if [ -z "${SKIP_TRAIN:-}" ]; then
  echo "==> run training (FAST_DEV_RUN=1)"
  docker run --rm \
    -e FAST_DEV_RUN=1 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    "$IMG_TRAIN"
fi

test -f checkpoints/classifier_v1.pt && echo "checkpoint present: checkpoints/classifier_v1.pt"

echo "==> run serving"
CID=$(docker run -d --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  "$IMG_SERVE")
trap 'docker stop "$CID" >/dev/null 2>&1 || true' EXIT

echo "==> wait for /health"
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8080/health >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS http://localhost:8080/health; echo

echo "==> write a 32x32 test image and call /predict"
base64 -d > test_image.png <<'PNG'
iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAATUlEQVR4nGN0y9vCQEvARFPTGUYtIAKMxgFBwIJLosmtg4FEULerAlNwNA4IgtEgIghGg4ggGA0igmA0iAY+iBhHG16EwGgyJQhoHkQAB9QGEV1sCTIAAAAASUVORK5CYII=
PNG
curl -fsS -X POST http://localhost:8080/predict -F "image=@test_image.png"; echo

echo "==> smoke test PASSED"
