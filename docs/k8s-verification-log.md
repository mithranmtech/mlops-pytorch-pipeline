# Verification log — Parts D, E, F (Kubernetes)

Cluster: `minikube v1.38.1` (docker driver), Kubernetes `v1.35.1`, single node.
Images built locally and loaded with `minikube image load mlops-train:v1 mlops-serve:v1`.

The training Job here runs with `SYNTHETIC_DATA=1` + `FAST_DEV_RUN=1` (the
commented block in `training-job.yaml`) so the end-to-end pass finishes in ~1 min
without the CIFAR-10 download. The committed manifest runs full CIFAR-10
training. Everything else — mounts, PVCs, resources, probes, Service, HPA,
rolling update — is exactly the committed manifests.

## Part D — training Job

```
$ kubectl apply -f k8s/namespace.yaml
$ kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml
namespace/ml-training created
configmap/training-config created
persistentvolumeclaim/training-data-pvc created
persistentvolumeclaim/model-checkpoints-pvc created

$ kubectl get pvc -n ml-training
NAME                    STATUS   VOLUME                CAPACITY   ACCESS MODES   STORAGECLASS
model-checkpoints-pvc   Bound    pvc-a95f3853-...      1Gi        RWO            standard
training-data-pvc       Bound    pvc-a24e0036-...      5Gi        RWO            standard

$ kubectl apply -f k8s/training-job.yaml       # + SYNTHETIC_DATA=1, FAST_DEV_RUN=1
$ kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=300s
job.batch/model-training condition met

$ kubectl get job model-training -n ml-training
NAME             STATUS     COMPLETIONS   DURATION   AGE
model-training   Complete   1/1           48s        48s

$ kubectl logs job/model-training -n ml-training
{"event": "config_loaded", "path": "/app/configs/training_config.yaml", "fast_dev_run": true}
{"event": "using_synthetic_data"}
{"event": "training_start", "device": "cpu", "epochs": 1, "params": 11173962}
{"epoch": 1, "train_loss": 2.3868, "train_accuracy": 0.0656, "val_loss": 4.1245, "val_accuracy": 0.0938, "seconds": 40.9}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "val_loss": 4.1245}
{"event": "training_complete", "best_val_loss": 4.1245, "checkpoint": "/app/checkpoints/classifier_v1.pt"}
```

Pod spec confirms the ConfigMap mount at `/app/configs`, PVCs at `/app/data` and
`/app/checkpoints`, and CPU 2 / Memory 4Gi requests+limits:

```
$ kubectl describe pod -n ml-training -l app=model-training
    Image:       mlops-train:v1
    Limits:      cpu: 2   memory: 4Gi
    Requests:    cpu: 2   memory: 4Gi
    Mounts:
      /app/checkpoints from checkpoints (rw)
      /app/configs from config (rw)
      /app/data from data (rw)
```

Checkpoint written to the shared PVC by the Job:

```
$ kubectl run pvc-check --rm -i --restart=Never --image=busybox -n ml-training \
    --overrides='{... mount model-checkpoints-pvc at /cp ...}' -- ls -la /cp
-rw-r--r--    1 root     root     134217690  classifier_v1.pt
```

## Part E — serving Deployment, Service, HPA

```
$ kubectl apply -f k8s/serving-deployment.yaml -f k8s/serving-service.yaml -f k8s/hpa.yaml
deployment.apps/model-serving created
service/model-serving created
horizontalpodautoscaler.autoscaling/model-serving created

$ kubectl rollout status deployment/model-serving -n ml-training
Waiting for deployment "model-serving" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "model-serving" rollout to finish: 1 of 2 updated replicas are available...
deployment "model-serving" successfully rolled out

$ kubectl get pods,svc,hpa -n ml-training
NAME                                 READY   STATUS      RESTARTS   AGE
pod/model-serving-74586655f7-gx522   1/1     Running     0          17s
pod/model-serving-74586655f7-mtxh6   1/1     Running     0          17s
pod/model-training-tsklt             0/1     Completed   0          74s

NAME                    TYPE        CLUSTER-IP      PORT(S)   AGE
service/model-serving   ClusterIP   10.106.199.29   80/TCP    17s

NAME                                                REFERENCE                  TARGETS              MINPODS   MAXPODS   REPLICAS
horizontalpodautoscaler.autoscaling/model-serving   Deployment/model-serving   cpu: <unknown>/70%   2         6         2
```
(`cpu: <unknown>` — metrics-server addon not enabled on this cluster; the HPA
object itself is correct: 2–6 replicas at 70% CPU.)

```
$ kubectl describe deployment model-serving -n ml-training
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
StrategyType:           RollingUpdate
RollingUpdateStrategy:  0 max unavailable, 1 max surge
    Image:      mlops-serve:v1
    Port:       8080/TCP (http)
    Limits:     cpu: 1      memory: 2Gi
    Requests:   cpu: 500m   memory: 1Gi
    Liveness:   http-get http://:8080/health delay=0s  timeout=1s period=10s failureThreshold=3
    Readiness:  http-get http://:8080/health delay=15s timeout=1s period=5s  failureThreshold=3
    Mounts:
      /app/checkpoints from checkpoints (ro)
      /app/configs from config (rw)
  Volumes:
   checkpoints:
    Type:       PersistentVolumeClaim
    ClaimName:  model-checkpoints-pvc
    ReadOnly:   true
```

## Part F — end-to-end prediction

```
$ kubectl port-forward svc/model-serving 8080:80 -n ml-training &

$ curl -s http://localhost:8080/health
{"status":"ok","model_loaded":true}

$ curl -s http://localhost:8080/
{"model_loaded":true,"architecture":"resnet18","dataset":"synthetic","num_classes":10,
 "trained_epochs":1,"val_accuracy":0.09375}

$ curl -s -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"class_0","confidence":0.719828,
 "probabilities":{"class_0":0.719828,"class_1":0.016532,"class_2":0.003395,"class_3":0.034096,
 "class_4":0.09639,"class_5":0.016525,"class_6":0.032817,"class_7":0.018459,"class_8":0.057866,
 "class_9":0.004092},
 "top_k":[{"class":"class_0","probability":0.719828}, ...]}
```

The serving pods loaded the checkpoint the Job wrote to the shared PVC
(`architecture: resnet18`, `dataset: synthetic` in the `/` response). With real
CIFAR-10 the class names would be `airplane`, `automobile`, … instead of
`class_0..9`, and the probabilities meaningful; on synthetic data the model has
learned nothing, which is expected.

## Rolling update (zero downtime)

```
$ kubectl rollout restart deployment/model-serving -n ml-training
deployment.apps/model-serving restarted

# availableReplicas during the rollout — never drops below 2 (maxUnavailable: 0)
3 total / 2 available / 1 updated
3 total / 2 available / 1 updated
3 total / 2 available / 2 updated
2 total / 2 available / 2 updated
rollout complete
```

## Summary

| Rubric item | Evidence |
|---|---|
| Namespace `ml-training` on every resource | `kubectl get all -n ml-training` |
| ConfigMap mounted at `/app/configs` | training pod `Mounts:` |
| PVCs for `/app/data` and `/app/checkpoints` | both `Bound`, mounted |
| Training Job requests+limits CPU 2 / Mem 4Gi | `describe pod` |
| Job produces the checkpoint on the PVC | `busybox` ls → `classifier_v1.pt` 134 MB |
| Serving: 2 replicas | `2/2 available` |
| Serving: checkpoint PVC read-only | `checkpoints (ro)`, `ReadOnly: true` |
| Liveness `/health` 10s ×3 | `describe deployment` |
| Readiness `/health` 5s, delay 15s | `describe deployment` |
| Requests 500m/1Gi, limits 1/2Gi | `describe deployment` |
| Rolling update maxSurge 1 / maxUnavailable 0 | `describe` + rollout watch |
| Service ClusterIP 80 → 8080 | `kubectl get svc` |
| HPA 2–6 @ 70% CPU | `kubectl get hpa` |
| End-to-end `/predict` via the Service | `curl` output above |
