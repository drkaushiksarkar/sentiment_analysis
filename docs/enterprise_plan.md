# Sentiment Intelligence Platform – Enterprise Upgrade Plan

## 1. Notebook Review Findings
- Colab-specific setup (`google.colab.drive` mounts, shell `!mkdir`, `os.chdir`) makes the code non-portable outside a single notebook runtime and prevents reproducible builds or CI runners (`NLP_Project.ipynb:62`, `NLP_Project.ipynb:94`, `NLP_Project.ipynb:128`, `NLP_Project.ipynb:139`, `NLP_Project.ipynb:2147`, `NLP_Project.ipynb:2158`).
- Dataset access is ad-hoc: IMDB reviews are loaded directly from `keras.datasets` without provenance, while sarcastic headlines are fetched from a public GitHub URL on every run, so there is no schema contract, caching, or versioning for compliance (`NLP_Project.ipynb:1540`).
- Models are defined inline (dense, CNN, bidirectional LSTM) with duplicated hyperparameters and minimal documentation, which makes it hard to trace experiments or reuse components (`NLP_Project.ipynb:555`, `NLP_Project.ipynb:1358`, `NLP_Project.ipynb:2955`, `NLP_Project.ipynb:3094`).
- Feature engineering truncates text to the first 20–32 tokens, leaving most semantic signal unused and causing the LSTM to underperform; there is no attention to multilingual or domain adaptation use cases (`NLP_Project.ipynb:170`, `NLP_Project.ipynb:1338`, `NLP_Project.ipynb:3290`).
- Deprecated APIs (`model.predict_proba`) are still invoked, emitting warnings and hinting at stale dependency management; there is no validation split management or automated metric tracking (`NLP_Project.ipynb:743`, `NLP_Project.ipynb:1478`).
- No tests, no environment specs, and no automation exist, so results cannot be trusted or promoted to production.

## 2. Target Experience & Requirements
1. **Use cases**: near-real-time sentiment & sarcasm scoring for digital content, analyst workflows for dataset curation, and experimentation with multiple transformer-based models.
2. **Non-functionals**: auditable data lineage, <200 ms P95 inference latency, multi-tenant RBAC, SOC2-ready logging, and zero-downtime deploys.
3. **Developer needs**: modular Python packages, typed React frontend, containerized services, IaC-managed infrastructure, and an LLM-powered agent that can plan & execute ML operations under human oversight.

## 3. Architecture Overview

### 3.1 Data & ML Platform
- **Ingestion**: Event-driven collectors (FastAPI workers + Kafka Connectors) pull raw posts, reviews, support tickets, etc. Batch backfills land in S3/ADLS Bronze buckets; metadata logged to a data catalog (OpenMetadata).
- **Processing & Feature Store**: Prefect or Dagster orchestrates cleaning, language detection, PII scrubbing, and topic tagging. Persist curated features in Feast backed by Redis + Snowflake to keep training/inference parity.
- **Experimentation**: Adopt PyTorch Lightning + Hugging Face Transformers for modern backbones (DistilRoBERTa, DeBERTaV3, IndicBERT for multilingual). Track runs and hyperparameters in MLflow; enforce structured config (Hydra) so experiments are reproducible.
- **Model Registry & Deployment**: Promote artifacts through MLflow stages (`Staging`, `Prod`). Package inference graphs behind a Triton or TorchServe deployment, wrapped by FastAPI for business logic. Store embeddings in a vector DB (PgVector or Pinecone) for semantic analytics.
- **Data Governance**: Version datasets with DVC or LakeFS, enforce schema via Great Expectations, and attach lineage to each model using MLflow tags.

### 3.2 Backend & Services (Python)
1. **Gateway (FastAPI + Pydantic v2)**: AuthN/AuthZ (OIDC/OAuth2), request quota, multi-tenant routing.
2. **Inference Service**: Async endpoints (`/api/v1/sentiment`, `/api/v1/sarcasm`) calling optimized transformer models on CPU/GPU via ONNX Runtime; supports batch scoring and streaming via WebSockets.
3. **Feedback & Labeling Service**: Stores user annotations, active-learning suggestions, and drift signals in Postgres; surfaces tasks to the frontend.
4. **Analytics/Reporting API**: Aggregates metrics, confusion matrices, A/B tests, uses ClickHouse for fast OLAP.
5. **Task Orchestrator**: Celery/Arq workers trigger training pipelines, evaluation sweeps, and Slack notifications; orchestrator exposes hooks for the AI agent.

### 3.3 Frontend (React)
- Stack: React 18 + TypeScript + Vite, TanStack Query for data fetching, Zustand or Redux Toolkit for complex state, Chakra UI / MUI for design system, Storybook for component QA.
- Key views:
  - **Real-time Monitoring**: streaming charts (Recharts + WebSockets) for latency, sentiment distribution, and drift alerts.
  - **Labeling Studio**: queue + detail panel + shortcut-friendly annotation controls, integrates with backend feedback API.
  - **Experiment Dashboard**: surfaces MLflow runs, confusion matrices, and allows triggering retrains (agent-mediated) with guard rails.
  - **Playground**: interactive text input, highlights explanation tokens (via Integrated Gradients/SHAP).
- Access control baked in via middleware + route guards; internationalization ready (react-intl).

### 3.4 Infrastructure & Ops
- **Containers**: Multi-stage Dockerfiles per service; base images pinned & scanned (Trivy).
- **Orchestration**: Deploy on Kubernetes (EKS/GKE) with Helm or Kustomize; use Argo Rollouts / Flagger for progressive delivery.
- **Secrets & Config**: Vault or AWS Secrets Manager; SSM Parameter Store for non-sensitive config.
- **Observability**: OpenTelemetry instrumentation -> Tempo/Jaeger (traces), Loki (logs), Prometheus + Grafana dashboards. Inference metrics emitted per tenant/model version.
- **Cost controls**: Cluster autoscaler, spot pools for non-critical batch jobs, GPU node pools isolated.

## 4. AI Agent Strategy
| Component | Purpose | Tooling | Human Control |
|-----------|---------|---------|----------------|
| **SentimentOps Planner** | LLM (GPT-4o Mini or open-source Llama 3.1 70B) that reads telemetry, backlog, and policies to draft experiment or deployment plans. | LangGraph / CrewAI with retrieval over design docs + run metadata. | Requires approval on every plan; surfaces diff + rationale in Slack.
| **Data Curation Agent** | Suggests labeling priorities, runs Great Expectations suites, files issues when schema drifts. | Uses Feast + GX APIs, interacts with Jira via webhooks. | Can open but not merge PRs; alerts data stewards for sign-off.
| **Training Executor Agent** | Converts approved plans into Hydra configs, triggers Prefect flows, tracks MLflow run IDs, and posts evaluations. | Access to CI artifacts, GPU queues via Kubernetes Jobs. | Guardrailed by policy engine (OPA); can only promote to `Staging` automatically.
| **Deployment Sentry** | Monitors prod metrics; if drift/latency thresholds breach, rolls back via Argo Rollouts or asks human to confirm blue/green switch. | Consumes Prometheus + Loki data; interacts with Argo API. | Human override required before scaling to zero or rolling back beyond one step.

Agents authenticate via short-lived service tokens, log every action, and must pass automated policy checks (OPA + Conftest). Implement red-teaming tests before granting prod access.

## 5. Testing & Quality Strategy
- **Python backend**: Pytest + coverage, property-based tests (Hypothesis) for tokenizer & inference pipelines, contract tests for Pydantic schemas, load tests with Locust targeting 99th percentile latency.
- **ML**: Unit tests for data transforms, smoke tests validating ONNX/TorchScript exports, golden datasets for regression, adversarial suites (typos, emojis, code-mixed text), bias/fairness reports (Fairlearn) stored per release.
- **Frontend**: Vitest + React Testing Library, Storybook interaction tests, Cypress e2e flows (auth, annotation, inference, admin settings), Lighthouse performance budgets.
- **Security**: SAST (Semgrep), dependency scanning (Dependabot + Snyk), container scans (Trivy), DAST (OWASP ZAP) in staging.
- **Observability tests**: OpenTelemetry trace assertions in CI to ensure spans & attributes exist before deploy.

## 6. Automation & CI/CD
1. **Lint + Type Checks**: `ruff`, `mypy`, `pyright`, `eslint`, `stylelint` across services.
2. **Unit Tests**: matrix across Python versions and frontend browsers (Playwright runners).
3. **ML Pipeline Checks**: kick off minimal Prefect flow with sampled data; verify metrics above floor.
4. **Build & Scan**: Docker images built with Buildx, signed (cosign), scanned.
5. **Deploy**: GitHub Actions -> ArgoCD; staging deploys auto, production requires manual approval + health checks. Canary analysis baked in (Kayenta or Flagger).
6. **Post-deploy**: Synthetic smoke tests, agent-based status update, rollback automation if KPIs degrade.

## 7. Implementation Roadmap
- **Phase 0 – Foundations (Week 0-2)**: Set up monorepo (pnpm + uv/pdm), define coding standards, scaffold React + FastAPI projects, add pre-commit hooks, configure CI skeleton.
- **Phase 1 – Data & Model Platform (Week 2-6)**: Build ingestion DAGs, create dataset schemas, integrate MLflow, stand up first transformer baseline (DistilRoBERTa) with evaluation harness + DVC-managed data snapshots.
- **Phase 2 – Backend & Frontend MVP (Week 4-8)**: Deliver core APIs (auth, inference, feedback) and React dashboards/playground; add real-time websockets and labeling UI.
- **Phase 3 – Automation & Agent (Week 6-10)**: Integrate Prefect + Celery workers, wire SentimentOps planner + training executor agents with policy guardrails, enable staged deployments via Argo.
- **Phase 4 – Hardening (Week 10-12)**: Load/perf testing, security scans, failover drills, finalize observability dashboards, document runbooks, and prep SOC2 evidence.

## 8. Immediate Next Steps
1. Export notebook logic into modular Python packages (data loaders, training scripts) and add environment specs (`pyproject.toml`, `requirements-lock.txt`).
2. Stand up repo structure (`apps/backend`, `apps/frontend`, `ml/`) with Docker + Makefile to unblock CI.
3. Draft detailed user stories for each frontend surface and align with stakeholders; feed them into the SentimentOps planner knowledge base.
4. Prioritize data governance setup (DVC + Great Expectations) so future models have trusted inputs.
