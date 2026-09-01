# Git workflow

Branch model: `main` ← `develop` ← `feature/*`. No direct commits to `main` or
`develop`. Every feature branch merges through a Pull Request with a real
description. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

`scripts/bootstrap_repo.sh` creates this history locally and pushes the four
feature branches. Then open the four PRs (titles + bodies below) and merge them
in order: `develop` first, then `develop → main`.

## Branch / PR plan

| # | Week | Branch | Scope |
|---|---|---|---|
| 1 | 1 | `feature/pytorch-model` | `src/`, `configs/`, `tests/`, `requirements/`, `pyproject.toml` |
| 2 | 1 | `feature/docker-training` | `docker/`, `.dockerignore`, `Makefile`, `scripts/`, `.github/workflows/ci.yml` |
| 3 | 2 | `feature/k8s-training-job` | `k8s/namespace.yaml`, `k8s/configmap.yaml`, `k8s/pvc.yaml`, `k8s/training-job.yaml` |
| 4 | 2 | `feature/k8s-serving` | `k8s/serving-deployment.yaml`, `k8s/serving-service.yaml`, `k8s/hpa.yaml`, `docs/`, `WRITEUP.md` |

---

## PR 1 — `feat: PyTorch CIFAR-10 classifier (model, data, training, serving)`

Base: `develop`

Implements Part B. Adds:

- `src/model.py` — `get_model()` returning a ResNet-18 with a CIFAR-adapted stem
  (3x3 stride-1 conv, no maxpool) or a lightweight `SimpleCNN`.
- `src/dataset.py` — CIFAR-10 / Fashion-MNIST loaders with train/eval transforms.
- `src/train.py` — config-driven loop; per-epoch metrics as JSON lines to stdout;
  early stopping; saves the best checkpoint (self-contained: arch, classes,
  normalization). `FAST_DEV_RUN=1` for a quick pass.
- `src/serve.py` — FastAPI app: `POST /predict` (multipart `image`), `GET /health`.
- `configs/training_config.yaml`, pinned `requirements/*.txt`, `pytest` suite.

Verification: `make lint test` green; `FAST_DEV_RUN=1 python src/train.py` writes
a checkpoint; `make serve` answers `/health` and `/predict`.

## PR 2 — `feat: Docker images for training and serving + CI`

Base: `develop`

Implements Part C. Adds:

- `docker/Dockerfile.train` — multi-stage (deps venv → clean slim runtime),
  `PYTHONUNBUFFERED=1`, config via mount / `$TRAINING_CONFIG_PATH`.
- `docker/Dockerfile.serve` — slim base, inference deps only (no `tensorboard`),
  non-root `appuser`, `EXPOSE 8080`, `HEALTHCHECK` on `/health`.
- `scripts/smoke_test.sh` — build both images, fast train, serve, curl `/predict`.
- `.github/workflows/ci.yml` — ruff, pytest, training smoke test, both image builds.

Verification: paste the `scripts/smoke_test.sh` terminal log (see the
"Verification" block in the assignment).

## PR 3 — `feat: Kubernetes training Job with ConfigMap and persistent storage`

Base: `develop`

Implements Part D. Adds `k8s/namespace.yaml`, `k8s/configmap.yaml`,
`k8s/pvc.yaml`, `k8s/training-job.yaml`. The Job mounts the ConfigMap at
`/app/configs`, PVCs at `/app/data` and `/app/checkpoints`, and sets
requests+limits of CPU 2 / Memory 4Gi. Commented GPU block for the bonus.

Verification: `kubectl apply` the three prerequisites + the Job, paste
`kubectl get job,pods -n ml-training` and `kubectl logs job/model-training`.

## PR 4 — `feat: Kubernetes serving Deployment, Service, HPA + end-to-end validation`

Base: `develop`

Implements Parts E & F. Adds `k8s/serving-deployment.yaml` (2 replicas,
read-only checkpoint PVC, liveness/readiness on `/health`, rolling update
`maxSurge 1 / maxUnavailable 0`), `k8s/serving-service.yaml` (ClusterIP
`80 → 8080`), `k8s/hpa.yaml`, and the reflection write-up.

Verification (put screenshots / logs in this PR body):
`kubectl get pods -n ml-training`, `kubectl describe deployment model-serving -n ml-training`,
`kubectl port-forward svc/model-serving 8080:80 -n ml-training` then
`curl -X POST http://localhost:8080/predict -F "image=@test_image.png"`.

---

## Manual commands (equivalent to the bootstrap script)

```bash
git init -b main
git add README.md .gitignore LICENSE docs/  # base
git commit -m "chore: initialize repository"
git branch develop

# --- PR 1 ---
git switch -c feature/pytorch-model develop
git add src configs tests requirements pyproject.toml
git commit -m "feat: PyTorch CIFAR-10 classifier with training and serving"
git push -u origin feature/pytorch-model
gh pr create --base develop --head feature/pytorch-model --fill

# repeat for feature/docker-training, feature/k8s-training-job, feature/k8s-serving
# then, after all four are merged into develop:
gh pr create --base main --head develop --title "release: week 1 + week 2" --fill
```

> If you use AI assistance for any code, cite it in that commit's body
> (`Assisted-by: <tool>`), per the assignment's academic-integrity note.
