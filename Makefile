.DEFAULT_GOAL := help
.PHONY: help install gpu-check lint format format-check typecheck test check \
        data prepare train evaluate repro exp exp-show push pull \
        predict app docker-build docker-up docker-down clean

IMAGE ?= rtsc-app
IMAGE_PATH ?=

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- environment -------------------------------------------------------------
install: ## Create .venv from uv.lock (incl. dev tools)
	uv sync --locked

gpu-check: ## Show torch version and CUDA availability
	uv run python check.py

# --- quality -----------------------------------------------------------------
lint: ## Ruff lint
	uv run ruff check .

format: ## Ruff autoformat
	uv run ruff format .
	uv run ruff check --fix .

format-check: ## Ruff format check (CI)
	uv run ruff format --check .

typecheck: ## Mypy
	uv run mypy

test: ## Unit tests
	uv run pytest

check: lint format-check typecheck test ## All quality gates

# --- pipeline (DVC) ----------------------------------------------------------
data: ## Download RTSD from Kaggle (if missing) and crop signs
	uv run dvc repro prepare

train: ## Train with params.yaml (runs data stages if needed)
	uv run dvc repro train

evaluate: ## Evaluate the current model
	uv run dvc repro evaluate

repro: ## Run the full pipeline
	uv run dvc repro

exp: ## Run a tracked experiment, e.g. make exp ARGS="-S train.lr=0.0005 -n lr5e-4"
	uv run dvc exp run $(ARGS)

exp-show: ## Compare experiments
	uv run dvc exp show --only-changed --md

push: ## Upload data/model versions to the DVC remote
	uv run dvc push

pull: ## Fetch data/model versions from the DVC remote (no retraining)
	uv run dvc pull

# --- usage / deploy ----------------------------------------------------------
predict: ## Predict one image: make predict IMAGE_PATH=path/to/sign.jpg
	uv run python src/predict.py --image $(IMAGE_PATH)

app: ## Run Streamlit demo locally
	uv run streamlit run src/app.py

docker-build: ## Build the inference image
	docker compose build

docker-up: ## Start the Streamlit app in Docker (GPU)
	docker compose up -d

docker-down: ## Stop the app
	docker compose down

clean: ## Remove caches (keeps data, models and DVC cache)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
