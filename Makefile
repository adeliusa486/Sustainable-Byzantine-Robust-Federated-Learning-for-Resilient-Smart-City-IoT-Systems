# =============================================================================
# AMFTA — Makefile
# =============================================================================
# Usage: make <target>
# Run:   make help

PYTHON     := python
PIP        := pip
PYTEST     := pytest
BLACK      := black
ISORT      := isort
FLAKE8     := flake8
UVICORN    := uvicorn
SOURCES    := amfta/ training/ evaluation/ experiments/ api/ tests/

.PHONY: help install install-dev lint format test test-unit test-integration \
        test-coverage api train preprocess partition smoke clean docker-build \
        docker-up docker-down docs figures

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:        ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:    ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint:           ## Run linters (flake8, mypy)
	$(FLAKE8) $(SOURCES) --max-line-length=95 --extend-ignore=E203,W503

format:         ## Auto-format code (black + isort)
	$(ISORT) $(SOURCES)
	$(BLACK) $(SOURCES)

format-check:   ## Check formatting without modifying files
	$(BLACK) --check $(SOURCES)
	$(ISORT) --check-only $(SOURCES)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:           ## Run all tests
	$(PYTEST) tests/ -v --timeout=120

test-unit:      ## Run unit tests only (fast)
	$(PYTEST) tests/test_trust_factors.py tests/test_model_and_data.py \
	         tests/test_aggregator.py -v --timeout=60

test-integration: ## Run integration tests (requires ~1 min)
	$(PYTEST) tests/test_integration.py -v --timeout=300

test-coverage:  ## Run tests with coverage report
	$(PYTEST) tests/ \
	  --cov=$(SOURCES) \
	  --cov-report=html:htmlcov \
	  --cov-report=term-missing \
	  --timeout=120
	@echo "HTML coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
preprocess:     ## Preprocess raw TON_IoT CSV data
	$(PYTHON) -m amfta.data.preprocessing \
	  --raw_dir data/raw \
	  --processed_dir data/processed \
	  --server_dir data/server

partition:      ## Partition training data for federated clients (Dirichlet α=0.5)
	$(PYTHON) -m amfta.data.partitioning \
	  --alpha 0.5 \
	  --num_clients 100 \
	  --partition_dir data/partitions

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
smoke:          ## Quick smoke test with synthetic data (no download required)
	$(PYTHON) experiments/run_main.py \
	  --method amfta \
	  --use_synthetic \
	  --num_rounds 5 \
	  --num_clients 10 \
	  --seeds 42 \
	  --byzantine_fraction 0.30 \
	  --log_level INFO

train:          ## Full training run (requires TON_IoT dataset)
	$(PYTHON) experiments/run_main.py --all

train-amfta:    ## Run AMFTA only (30% Byzantine, label flipping)
	$(PYTHON) experiments/run_main.py \
	  --method amfta \
	  --byzantine_fraction 0.30 \
	  --attack label_flipping

ablation:       ## Run ablation study
	$(PYTHON) experiments/run_ablation.py --use_synthetic --num_rounds 20

figures:        ## Generate all figures from saved results
	$(PYTHON) evaluation/visualize.py \
	  --results_dir results \
	  --output_dir figures

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
api:            ## Start inference API (development mode)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:       ## Start inference API (production mode, 4 workers)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 4

api-test:       ## Test running API with a sample prediction
	curl -s -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, \
	                    0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, \
	                    0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, \
	                    0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, \
	                    0.1, 0.2, 0.3, 0.4, 0.5]}' | python -m json.tool

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:   ## Build Docker image
	docker build -t amfta-fl:latest .

docker-up:      ## Start all Docker services
	docker-compose up -d

docker-down:    ## Stop all Docker services
	docker-compose down

docker-test:    ## Run tests inside Docker
	docker-compose run --rm amfta-test

docker-train:   ## Run quick training inside Docker
	docker-compose run --rm amfta-train

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:          ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete
	@echo "Cleaned build artifacts."

clean-results:  ## Remove experiment results (CAREFUL: destructive)
	rm -rf results/* figures/* checkpoints/*
	@echo "Cleaned results, figures, and checkpoints."
