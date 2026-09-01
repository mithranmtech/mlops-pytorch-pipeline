#!/usr/bin/env bash
# Build the main <- develop <- feature/* history described in docs/git-workflow.md
# and push the four feature branches. Does NOT open or merge PRs -- do that on
# GitHub so each merge is a reviewed Pull Request.
#
# Usage:
#   scripts/bootstrap_repo.sh git@github.com:<you>/mlops-pytorch-pipeline.git
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-}"
[ -n "$REMOTE" ] || { echo "usage: $0 <git-remote-url>"; exit 1; }
[ -d .git ] && { echo ".git already exists; refusing to re-bootstrap"; exit 1; }

commit() { git commit -q -m "$1" ${2:+-m "$2"}; echo "  committed: $1"; }

git init -q -b main
git remote add origin "$REMOTE"

# ---- base commit on main ------------------------------------------------
git add README.md .gitignore LICENSE docs/git-workflow.md
commit "chore: initialize repository structure"
git branch develop

# ---- PR 1: model / data / training / serving --------------------------
git switch -q -c feature/pytorch-model develop
git add src configs tests requirements pyproject.toml WRITEUP.md
commit "feat: PyTorch CIFAR-10 classifier with training and serving" \
       "CNN/ResNet-18 model, torchvision data pipeline, config-driven training
loop with JSON-lines metrics and early stopping, FastAPI server exposing
/predict and /health, pinned requirements, pytest suite."

# ---- PR 2: docker + CI ------------------------------------------------
git switch -q -c feature/docker-training develop
git add docker .dockerignore Makefile scripts .github docs/verification-log.md
commit "feat: multi-stage Docker images for training and serving + CI" \
       "Dockerfile.train is multi-stage; Dockerfile.serve is slim, inference-only,
non-root, with a HEALTHCHECK. Adds smoke_test.sh, a GitHub Actions workflow
(ruff, pytest, training smoke test, image builds), and a captured
end-to-end verification log."

# ---- PR 3: k8s training job -----------------------------------------
git switch -q -c feature/k8s-training-job develop
git add k8s/namespace.yaml k8s/configmap.yaml k8s/pvc.yaml k8s/training-job.yaml
commit "feat: Kubernetes training Job with ConfigMap and PVC storage" \
       "ml-training namespace, training-config ConfigMap mounted at /app/configs,
PVCs for /app/data and /app/checkpoints, CPU 2 / Memory 4Gi requests+limits,
commented GPU block for the bonus."

# ---- PR 4: k8s serving --------------------------------------------
git switch -q -c feature/k8s-serving develop
git add k8s/serving-deployment.yaml k8s/serving-service.yaml k8s/hpa.yaml
commit "feat: Kubernetes serving Deployment, Service and HPA" \
       "2-replica Deployment, read-only checkpoint PVC, liveness/readiness probes
on /health, rolling update maxSurge 1 / maxUnavailable 0, ClusterIP Service
80->8080, HPA 2-6 replicas at 70% CPU."

git switch -q develop
git push -u origin main
git push -u origin develop
for b in feature/pytorch-model feature/docker-training feature/k8s-training-job feature/k8s-serving; do
  git push -u origin "$b"
done

cat <<'EOF'

Branches pushed. Now on GitHub:
  1. Open PR: feature/pytorch-model    -> develop   (use body from docs/git-workflow.md)
  2. Open PR: feature/docker-training   -> develop
  3. Open PR: feature/k8s-training-job  -> develop
  4. Open PR: feature/k8s-serving       -> develop   (attach validation screenshots)
  Merge all four, then open PR: develop -> main and merge.
EOF
