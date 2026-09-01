# Verification log — Part C (Docker)

Captured from `./scripts/smoke_test.sh` on macOS / Docker Desktop 29.2.1
(`linux/arm64`). Paste this (or a fresh run) into the PR 2 description.

## Images build

```
==> build training image
#8 [base 4/4] RUN python -m venv /opt/venv && pip install -r requirements/train.txt --extra-index-url https://download.pytorch.org/whl/cpu
#8 Successfully installed PyYAML-6.0.2 ... numpy-2.1.3 ... tensorboard-2.18.0 torch-2.5.1 torchvision-0.20.1 ...
#8 DONE 107.7s
#9 [training 3/5] COPY --from=base /opt/venv /opt/venv        <- multi-stage: only the venv crosses
#12 naming to docker.io/library/mlops-train:v1 done

==> build serving image
#9 [base 4/4] RUN ... pip install -r requirements/serve.txt ...
#9 Successfully installed ... fastapi-0.115.5 ... torch-2.5.1 torchvision-0.20.1 uvicorn-0.32.1 ...   (no tensorboard)
#13 [serving 7/7] RUN mkdir -p /app/checkpoints && chown -R appuser:appuser /app
#14 naming to docker.io/library/mlops-serve:v1 done
```

## Training container (mounted volumes, config from /app/configs)

```
==> run training (FAST_DEV_RUN=1)
docker run --rm -e FAST_DEV_RUN=1 -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1
{"event": "config_loaded", "path": "/app/configs/training_config.yaml", "fast_dev_run": true}
Files already downloaded and verified
{"event": "training_start", "device": "cpu", "epochs": 1, "params": 11173962}
{"epoch": 1, "train_loss": 2.4398, "train_accuracy": 0.1313, "val_loss": 2.6505, "val_accuracy": 0.1938, "seconds": 622.2}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "val_loss": 2.6505}
{"event": "training_complete", "best_val_loss": 2.6505, "checkpoint": "/app/checkpoints/classifier_v1.pt"}
checkpoint present: checkpoints/classifier_v1.pt
```
(`FAST_DEV_RUN=1` caps it at 1 epoch / 5 batches. Drop it for a full 10-epoch run.)

## Serving container (checkpoint from mounted volume)

```
==> run serving
docker run -d --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1

$ curl -fsS http://localhost:8080/health
{"status":"ok","model_loaded":true}

$ curl -fsS -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"airplane","confidence":0.212101,
 "probabilities":{"airplane":0.212101,"automobile":0.050232,"bird":0.063532,"cat":0.049706,
 "deer":0.165476,"dog":0.080793,"frog":0.045302,"horse":0.139484,"ship":0.098112,"truck":0.095262},
 "top_k":[{"class":"airplane","probability":0.212101}, ...]}
==> smoke test PASSED
```
(Probabilities are near-uniform because `FAST_DEV_RUN` trains on ~320 images.)

## Requirement checks

```
$ docker inspect mlops-serve:v1 --format 'User={{.Config.User}} Exposed={{json .Config.ExposedPorts}} HC={{json .Config.Healthcheck.Test}}'
User=appuser  Exposed={"8080/tcp":{}}  HC=["CMD-SHELL","python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status == 200 else 1)\""]

$ docker run --rm mlops-serve:v1 id
uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)

$ docker run --rm mlops-serve:v1 pip list | grep -i tensorboard
(no output — training-only libs excluded from the serving image)
```

| Rubric item | Evidence |
|---|---|
| `Dockerfile.train` multi-stage | `base` deps stage + `training` runtime stage, `COPY --from=base /opt/venv` |
| pinned deps | `torch==2.5.1`, `torchvision==0.20.1`, ... resolved exactly |
| config from mount / env | `TRAINING_CONFIG_PATH` + `/app/configs/training_config.yaml` |
| serve: slim base | `python:3.11-slim` |
| serve: inference deps only | no `tensorboard` in `pip list` |
| serve: EXPOSE 8080 | `Exposed={"8080/tcp":{}}` |
| serve: non-root | `uid=1000(appuser)` |
| serve: HEALTHCHECK | present, hits `/health` |
