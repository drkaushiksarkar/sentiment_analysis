# Sentiment Frontend

Vite + React + TypeScript scaffold for the sentiment platform UI.

## Local Development

```bash
cd apps/frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/api` requests to `http://localhost:8000`.

## Production Build

```bash
npm run build
npm run preview
```

## Docker

```bash
docker build -t sentiment-frontend:dev -f apps/frontend/Dockerfile .
docker run --rm -p 4173:80 sentiment-frontend:dev
```

The container serves the compiled assets via nginx.
