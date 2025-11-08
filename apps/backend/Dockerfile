# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY apps/backend/pyproject.toml apps/backend/README.md ./
COPY apps/backend/src ./src

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "backend_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
