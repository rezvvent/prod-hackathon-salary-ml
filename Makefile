PYTHON := .venv/bin/python

.PHONY: help data eda train eval onnx pipeline test api bench patch-p1 patch-p1-thresh patch-p2 patch-p3 docker-build docker-up

help:
	@echo "Available commands for Hackathon ML Pipeline:"
	@echo "  make data          - Generate market dataset, clean, filter anomalies, and create group splits"
	@echo "  make eda           - Run EDA, create statistical hypotheses and reports/eda.png"
	@echo "  make train         - Train Baselines, Quantile CatBoost (Q10/Q50/Q90), and log to MLflow"
	@echo "  make eval          - Evaluate on frozen hold-out test set with subgroup breakdown"
	@echo "  make onnx          - Export CatBoost models to ONNX and benchmark latency"
	@echo "  make pipeline      - Run entire pipeline (data -> eda -> train -> eval -> onnx) with 1 command"
	@echo "  make test          - Run pytest test suite"
	@echo "  make api           - Start FastAPI inference server on port 8000"
	@echo "  make bench         - Run async stress test (p50/p95/p99 latency & throughput)"
	@echo "  make patch-p1      - Mini-pravka P1: Add interaction feature on the fly"
	@echo "  make patch-p1-thresh - Mini-pravka P1: Change quantile thresholds (Q25-Q75)"
	@echo "  make patch-p2      - Mini-pravka P2: Launch new tracked experiment in MLflow"
	@echo "  make patch-p3      - Mini-pravka P3: Live audit logs & ONNX latency demo"
	@echo "  make docker-build  - Build production Docker image"
	@echo "  make docker-up     - Start containerized service via Docker Compose"

data:
	$(PYTHON) -m src.data_generator
	$(PYTHON) -m src.data_loader

eda:
	$(PYTHON) -m src.eda

train:
	$(PYTHON) -m src.train

eval:
	$(PYTHON) -m src.evaluate

onnx:
	$(PYTHON) -m src.onnx_export

pipeline: data eda train eval onnx
	@echo "Full reproducible ML Pipeline finished successfully!"

test:
	$(PYTHON) -m pytest -v tests/

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

bench:
	$(PYTHON) -m benchmark.load_test

patch-p1:
	$(PYTHON) -m live_patches.patch_p1_add_feature

patch-p1-thresh:
	$(PYTHON) -m live_patches.patch_p1_change_quantiles

patch-p2:
	$(PYTHON) -m live_patches.patch_p2_new_experiment

patch-p3:
	$(PYTHON) -m live_patches.patch_p3_show_logs_and_load

docker-build:
	docker build -t salary-ml-service:latest .

docker-up:
	docker compose up --build -d
