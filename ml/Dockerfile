# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY ml/requirements.txt ml/scripts/ ./ml/

RUN pip install --upgrade pip && pip install -r ml/requirements.txt

CMD ["bash", "ml/scripts/train_imdb.sh"]
