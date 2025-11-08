# Sentiment Backend

Lightweight FastAPI gateway that exposes health probes and stubs for inference endpoints. It is designed to evolve into the production API tier.

## Local Development

```bash
cd apps/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
uvicorn backend_app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Testing

```bash
pytest
```

## Docker

```bash
docker build -t sentiment-backend:dev -f apps/backend/Dockerfile .
docker run --rm -p 8000:8000 sentiment-backend:dev
```

The container uses `uvicorn` with hot reload disabled; edit `Dockerfile` if you need different workers or logging configuration.
