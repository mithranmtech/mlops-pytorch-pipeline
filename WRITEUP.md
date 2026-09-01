# Reflection

## What was the most challenging part?

The hardest part was not the model — a CIFAR ResNet-18 is a solved problem — but
making a **single artifact behave correctly in three very different runtimes**:
a bare `python src/train.py` from the repo root, `docker run` with bind mounts,
and a Kubernetes pod with a ConfigMap volume and PVCs. Each one presents the
config file, the dataset directory, and the checkpoint directory at a different
path. The fix was to stop hard-coding paths and define one resolution rule
(`$TRAINING_CONFIG_PATH` → `/app/configs/...` → `configs/...`) that every entry
point shares, plus keeping the local `configs/training_config.yaml` and the
`ConfigMap` in `k8s/configmap.yaml` byte-for-byte aligned on schema. Once that
contract was explicit, the Docker and Kubernetes layers became mechanical.

The second sharp edge was the **training/serving handoff**. Training and serving
never share a process; they only share a volume. That means the checkpoint has
to be self-describing — I save the architecture name, class list, and
normalization statistics inside the `.pt` file so `serve.py` can rebuild the
exact model and preprocessing without reading the training config at all. It
also forced a decision about storage: `ReadWriteOnce` is all a single-node
cluster offers, so two serving replicas mounting the checkpoint PVC only works
because they co-locate; a multi-node deployment needs `ReadWriteMany` or the
checkpoint baked into the image. Naming that limitation in the manifests felt
more honest than pretending it scales.

Third, the **Docker image size / correctness tradeoff**. `pip install torch` on
`python:3.11-slim` pulls the multi-gigabyte CUDA build by default. Pointing pip
at the CPU wheel index and using a multi-stage build (install into a venv in
stage one, copy only the venv into a clean stage two) cut the training image
substantially, and the serving image drops `tensorboard` and the training code
entirely. Getting the `HEALTHCHECK` right on a `curl`-less slim image meant
falling back to a one-line `urllib` probe.

## What I would do next

Add a real dataset cache (an init container that pre-downloads CIFAR-10 into the
data PVC so the Job doesn't re-download on every run), push images to a registry
so the manifests work on any cluster without `minikube image load`, and wire the
`FAST_DEV_RUN` smoke path into a nightly GitHub Action that also runs `kubectl
apply` against a `kind` cluster for true end-to-end CI.
