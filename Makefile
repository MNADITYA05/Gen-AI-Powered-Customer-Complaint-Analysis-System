.PHONY: install dev api frontend test lint clean docker-up docker-down generate train

# ─── Python interpreter ───────────────────────────────────────────────────────
# Using `python3 -m pip` ensures pip and python always refer to the same
# interpreter, avoiding conflicts between conda / system / pyenv installations.
# Override with: make install PYTHON=python3.11
PYTHON ?= python3

# ─── Setup ───────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -e ".[dev]"

env:
	cp .env.example .env
	@echo ".env created — fill in secrets before running."

# ─── Run ─────────────────────────────────────────────────────────────────────
api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501

dev:
	@echo "Starting API and frontend in parallel..."
	@make api & make frontend

# ─── Docker ──────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

# ─── Tests ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --cov=core --cov=api --cov-report=term-missing

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# ─── Code quality ────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check . --fix

format:
	$(PYTHON) -m ruff format .

# ─── Data & ML shortcuts ─────────────────────────────────────────────────────
generate:
	@echo "POST to /api/v1/complaints/generate or use the Streamlit UI."

migrate-db:
	$(PYTHON) -c "from core.database import create_tables; create_tables(); print('Tables created.')"

# ─── CFPB real dataset ───────────────────────────────────────────────────────
# 1. Download the raw CFPB CSV from:
#      https://www.consumerfinance.gov/data-research/consumer-complaints/
#    Save it as data/cfpb_raw.csv
# 2. Run: make prepare-cfpb   (maps columns, samples 5000 rows)
# 3. Run: make seed-cfpb      (inserts into DB)
# 4. Run: make api  →  Admin Train page  →  Train

# ─── Gen AI — transformer + RAG dependencies ─────────────────────────────────
# Run once to install all Gen AI deps (multi-task transformer + RAG).
install-genai:
	$(PYTHON) -m pip install transformers torch sentence-transformers faiss-cpu

# Rebuild the FAISS similarity index from DB (run after bulk imports).
rebuild-rag:
	$(PYTHON) -c "\
import os; os.environ.setdefault('DATABASE_URL','sqlite:///./data/complaints.db'); \
os.environ.setdefault('MODEL_DIR','models'); \
os.environ.setdefault('MLFLOW_TRACKING_URI','sqlite:///./mlruns.db'); \
os.environ.setdefault('SEEDS_DIR','data/seeds'); \
from core.database import SessionLocal; from core.rag_engine import get_rag_engine; \
db=SessionLocal(); e=get_rag_engine(); n=e.rebuild_index(db); db.close(); print(f'Indexed {n} complaints')"

# ─── Model benchmark ─────────────────────────────────────────────────────────
# Installs benchmark deps then runs full model comparison.
# Results saved to data/benchmark_results.csv
install-benchmark:
	$(PYTHON) -m pip install -e ".[benchmark]"

benchmark:
	$(PYTHON) scripts/benchmark_models.py --input data/cfpb_prepared.csv --output data/benchmark_results.csv

# ─── CFPB real dataset ───────────────────────────────────────────────────────
prepare-cfpb:
	$(PYTHON) scripts/prepare_cfpb_data.py --input data/cfpb_raw.csv --output data/cfpb_prepared.csv --limit 5000

seed-cfpb:
	$(PYTHON) scripts/seed_from_cfpb.py --input data/cfpb_prepared.csv

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage

clean-models:
	rm -rf models/ mlruns/

clean-data:
	rm -rf data/exports/* data/uploads/*
