.PHONY: help install lint test train serve docker-train docker-serve k8s-deploy k8s-clean smoke

IMG_TRAIN ?= mlops-train:v1
IMG_SERVE ?= mlops-serve:v1
NS         ?= ml-training

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install dev dependencies (CPU torch)
	pip install -r requirements/dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

lint:  ## Run ruff
	ruff check .

test:  ## Run unit tests
	pytest -v

train:  ## Train locally (writes ./checkpoints/classifier_v1.pt)
	TRAINING_CONFIG_PATH=configs/training_config.yaml \
	  python -c "import yaml,pathlib; p=pathlib.Path('configs/training_config.yaml'); c=yaml.safe_load(p.read_text()); c['data']['data_dir']='data'; c['output']['checkpoint_dir']='checkpoints'; p.write_text(yaml.safe_dump(c))"
	mkdir -p data checkpoints
	python src/train.py

serve:  ## Serve locally on :8080
	CHECKPOINT_PATH=checkpoints/classifier_v1.pt python src/serve.py

docker-train:  ## Build the training image
	docker build -f docker/Dockerfile.train -t $(IMG_TRAIN) .

docker-serve:  ## Build the serving image
	docker build -f docker/Dockerfile.serve -t $(IMG_SERVE) .

smoke: docker-train docker-serve  ## Full local Docker round-trip (see scripts/smoke_test.sh)
	./scripts/smoke_test.sh

k8s-deploy:  ## Apply all manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml
	kubectl apply -f k8s/training-job.yaml
	@echo "wait for the Job to complete, then:"
	@echo "  kubectl apply -f k8s/serving-deployment.yaml -f k8s/serving-service.yaml -f k8s/hpa.yaml"

k8s-clean:  ## Delete the namespace and everything in it
	kubectl delete namespace $(NS) --ignore-not-found
