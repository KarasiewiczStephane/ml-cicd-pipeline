# ML CI/CD Pipeline

![Build Status](https://github.com/KarasiewiczStephane/ml-cicd-pipeline/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)
![Model Accuracy](https://img.shields.io/badge/accuracy-96.7%25-brightgreen)
![Last Trained](https://img.shields.io/badge/last%20trained-2026--02--27-blue)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

> GitHub Actions workflows automating the full ML lifecycle: testing, data validation, training, evaluation, deployment, and rollback.

## Features

- Automated CI with linting (ruff), testing (pytest), and coverage gates (80%+)
- Data validation pipeline with schema, distribution, and missing-value checks
- Model training with scikit-learn, performance gates, and production comparison
- Versioned model artifacts via GitHub Releases
- Training cache to skip redundant retraining
- Docker-based deployment to GitHub Container Registry
- Automated rollback on deployment failures with issue alerts
- Multi-environment configs (dev, staging, prod) with escalating thresholds
- Dynamic badges and metrics history tracking with trend visualization

## Architecture

```mermaid
graph LR
    A[Push] --> B[CI Pipeline]
    B --> C{Tests Pass?}
    C -->|Yes| D[Data Validation]
    D --> E[Model Training]
    E --> F{Performance Gate}
    F -->|Pass| G[Docker Build]
    G --> H[Deploy Staging]
    H --> I{Integration Tests}
    I -->|Pass| J[Deploy Production]
    I -->|Fail| K[Rollback]
    K --> L[Create Issue]
```

## Quick Start

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/ml-cicd-pipeline.git
cd ml-cicd-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the full ML pipeline
make run

# Run tests
make test

# Lint and format
make lint

# Run with coverage
make coverage
```

## Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| [CI](.github/workflows/ci.yml) | Push / PR | Lint, test, coverage gate (Python 3.10-3.12) |
| [Data Validation](.github/workflows/data-validation.yml) | Data changes / Manual | Schema, distribution, quality checks |
| [Training](.github/workflows/train.yml) | Manual / Schedule / Data change | Train, evaluate, performance gate, release |
| [Deploy](.github/workflows/deploy.yml) | After training | Build Docker, deploy staging/prod, rollback |
| [Badge Update](.github/workflows/badge-update.yml) | After CI / Training | Update README badges |

## Project Structure

```
ml-cicd-pipeline/
├── .github/workflows/       # CI/CD pipeline definitions
│   ├── ci.yml               # Lint, test, coverage
│   ├── data-validation.yml  # Data quality checks
│   ├── train.yml            # Model training + evaluation
│   ├── deploy.yml           # Docker build, staging, production
│   └── badge-update.yml     # Dynamic badge updates
├── src/
│   ├── data/
│   │   ├── loader.py        # Iris dataset loading and splitting
│   │   └── validator.py     # Schema, distribution, quality validation
│   ├── models/
│   │   ├── train.py         # sklearn pipeline training
│   │   ├── evaluate.py      # Metrics computation and reporting
│   │   └── registry.py      # Production model registry + perf gate
│   ├── deploy/
│   │   └── health_check.py  # Deployment health verification
│   ├── utils/
│   │   ├── config.py        # YAML config loader (multi-env)
│   │   ├── cache.py         # Training cache (content hashing)
│   │   ├── badge_generator.py   # shields.io badge URLs
│   │   ├── metrics_tracker.py   # Metrics history + trend charts
│   │   ├── release_notes.py     # Model release notes generator
│   │   └── deployment_tracker.py # Deployment version history
│   └── main.py              # Pipeline orchestration entry point
├── tests/
│   ├── unit/                # Unit tests for all modules
│   └── integration/         # End-to-end deployment tests
├── configs/
│   ├── config.yaml          # Default configuration
│   ├── dev.yaml             # Development (threshold: 0.80)
│   ├── staging.yaml         # Staging (threshold: 0.85)
│   └── prod.yaml            # Production (threshold: 0.90)
├── Dockerfile               # Production container with HEALTHCHECK
├── Makefile                 # Common development commands
├── pyproject.toml           # Ruff + pytest config
├── requirements.txt         # Python dependencies
└── .pre-commit-config.yaml  # Pre-commit hooks (ruff, whitespace)
```

## Environment Configuration

Each environment has its own YAML config with appropriate thresholds:

| Environment | Accuracy Threshold | Debug | Log Level |
|-------------|-------------------|-------|-----------|
| `dev`       | 0.80              | true  | DEBUG     |
| `staging`   | 0.85              | false | INFO      |
| `prod`      | 0.90              | false | WARNING   |

Set the environment via the `ENVIRONMENT` variable:

```bash
ENVIRONMENT=staging python -m src.main
```

## How It Works

1. **CI Pipeline** runs on every push: ruff linting, pytest with coverage gate at 80%, matrix testing across Python 3.10/3.11/3.12
2. **Data Validation** checks schema, missing values, row counts, and distribution drift (KS test)
3. **Training Pipeline** trains a RandomForest classifier on the Iris dataset, evaluates metrics, and compares against the production model
4. **Performance Gate** blocks promotion if accuracy drops below the threshold or regresses vs. production
5. **Deployment** builds a Docker image, pushes to GHCR, tests in staging, then promotes to production
6. **Rollback** automatically reverts to the previous image and creates a GitHub issue if staging tests fail
7. **Metrics Tracking** appends metrics to a history file and generates trend charts

## Troubleshooting

**Tests failing locally?**
```bash
pip install -r requirements.txt
pytest tests/ -v --tb=long
```

**Pre-commit hooks failing?**
```bash
pre-commit install
pre-commit run --all-files
```

**Need to retrain the model?**
```bash
python -m src.models.train
python -m src.models.evaluate
```

## License

MIT
