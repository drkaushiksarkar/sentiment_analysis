# Sentiment Intelligence Platform

[![CI](https://github.com/drkaushiksarkar/sentiment_analysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/drkaushiksarkar/sentiment_analysis/actions/workflows/ci.yml?query=branch%3Amain)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-production--ready-success)
![React](https://img.shields.io/badge/React-18.x-61dafb)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Enterprise-ready sentiment analytics platform that pairs modern NLP models with a real‑time React dashboard, CI/CD automation, and production-friendly service scaffolding.

---

## Key Capabilities

- **Modular ML package** (`sentiment_package`) offering reusable data loaders, model builders, and training APIs for IMDB sentiment and sarcasm detection.  
- **FastAPI backend** with health probes, model-backed inference (`/api/v1/sentiment`, `/api/v1/sentiment/batch`) and analytics endpoints (`/api/v1/metrics/sentiment`) that expose rolling confidence/timeline data.  
- **React + Vite frontend** featuring live health cards, inference playground, and sentiment performance charts.  
- **Automation ready**: GitHub Actions workflow enforces linting (`ruff`, `eslint`) and test suites (`pytest`, frontend build) on every push/PR.  
- **Container & IaC friendly**: Dockerfiles for backend and frontend, ML Docker base for batch training, and explicit dependency metadata (pyproject / package.json).  
- **Enterprise hygiene**: `.gitattributes` tuned so notebooks are ignored in language stats, code formatted/linted, and README documents operational runbooks.

---

## Repository Map

```
├── apps
│   ├── backend/        # FastAPI service, tests, Dockerfile, pyproject
│   └── frontend/       # Vite + React dashboard, npm scripts, Dockerfile
├── ml/                 # Training Dockerfile + scripts that call the shared package
├── scripts/            # CLI entry points (train_imdb.py, train_sarcasm.py)
├── src/sentiment_package/
│   ├── imdb/           # Data, models, training utilities for IMDB reviews
│   └── sarcasm/        # Sarcasm dataset utilities + GloVe integration
├── artifacts/          # Model checkpoints (not committed)
├── tests/              # Package-level smoke/shape tests
└── .github/workflows/  # CI pipeline definition
```

---

## Tech Stack

| Layer      | Technologies / Notes |
|------------|----------------------|
| **ML**     | TensorFlow/Keras, PyTorch-ready scaffolding, Pytest, DVC-friendly structure |
| **Backend**| FastAPI, Pydantic v2, Uvicorn, HTTPX test client, extensible service modules |
| **Frontend**| React 18, TypeScript, Vite, ESLint, lightweight charting via SVG |
| **Automation**| GitHub Actions, Ruff, Pytest, npm lint/build, Docker multi-stage builds |

---

## Getting Started

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pip install -e "apps/backend[dev]"
```

### 2. Frontend dependencies

```bash
cd apps/frontend
npm install
```

### 3. Model artifacts

1. Train (or download) the dense IMDB model:
   ```bash
   source .venv/bin/activate
   python scripts/train_imdb.py --model dense --epochs 4 --checkpoint-dir artifacts/imdb_dense
   ```
2. Obtain `imdb_word_index.json` from a trusted source (or another machine with internet access) and set  
   `SENTIMENT_BACKEND_IMDB_WORD_INDEX_PATH=/absolute/path/imdb_word_index.json`.  
   Place the file under `~/.keras/datasets/` to reuse across environments. Without it the backend automatically falls back to a heuristic classifier so development remains unblocked.

### 4. Run the services

Backend (FastAPI + TensorFlow model):
```bash
source .venv/bin/activate
uvicorn backend_app.main:app --app-dir apps/backend/src --port 8000
```

Frontend (React dashboard):
```bash
cd apps/frontend
npm run dev
```

Visit `http://localhost:5173` – the dashboard proxies `/api/*` to `http://localhost:8000`, showing health probes, sentiment playground, and live metrics.

---

## CI / Quality Gates

- `pytest` (root): runs package smoke tests plus backend API suites.  
- `pytest apps/backend`: backend-specific tests; automatically exercised by the CI workflow.  
- `npm run lint && npm run build` inside `apps/frontend`: ensures UI integrity before merge.  
- GitHub Actions workflow (`.github/workflows/ci.yml`) provisions Python 3.11, installs editable packages, runs Ruff + Pytest, builds the frontend, and blocks regressions on PRs.

---

## Operational Notes

- **Artifacts**: Keep model weights under `artifacts/` (git-ignored). Point `SENTIMENT_BACKEND_IMDB_WEIGHTS_PATH` or override the default path if needed.  
- **Telemetry / Charts**: The backend’s `StatsTracker` stores recent in-memory stats. Extend it with Redis/Prometheus exporters for multi-instance deployments.  
- **Docker**: `apps/backend/Dockerfile` and `apps/frontend/Dockerfile` ship production-ready images. `ml/Dockerfile` powers batch training jobs.  
- **Vendored notebooks**: `.gitattributes` marks `*.ipynb` as `linguist-vendored` to keep GitHub language stats focused on the production codebase.

---

## Roadmap Ideas

- Swap the heuristic fallback with a locally cached tokenizer JSON to remove external dependencies completely.  
- Add Prometheus exporters + Grafana dashboards for latency/confidence metrics.  
- Integrate Argo Workflows / Prefect for reproducible training orchestrations.  
- Expand the frontend with latency histograms and data quality alerts once telemetry endpoints are available.

---

© Sentiment Intelligence Platform – MIT License. Built for enterprise-grade NLP experimentation and deployment.***
