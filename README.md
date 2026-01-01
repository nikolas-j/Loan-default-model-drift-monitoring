# Production-Grade ML System: Loan Default Prediction

[![CI - Tests and MLflow Integration](https://github.com/nikolas-j/Loan-default-model-drift-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/nikolas-j/Loan-default-model-drift-monitoring/actions/workflows/ci.yml)

An end-to-end machine learning system demonstrating production-ready practices for credit risk assessment. The system predicts loan defaults using Gradient Boosting Classifier with comprehensive experiment tracking, automated testing, drift monitoring, and containerized deployment.

**Dataset**: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Key Features

This project demonstrates the following ML engineering practices:

- **MLflow Experiment Tracking & Model Registry**: Full experiment logging with versioned models and @staging / @production alias for deployment
- **Automated CI/CD Pipeline**: GitHub Actions workflow running unit and integration tests on every commit
- **Drift Monitoring**: Population Stability Index (PSI) for feature drift detection and binomial 3-sigma control limits for model performance monitoring
- **Containerized API**: Docker-based FastAPI service serving production models
- **Production Code Quality**: Centralized configuration, structured logging, comprehensive unit/integration tests
- **Data Validation**: Schema validation and feature engineering pipeline

## Architecture

![Architecture flow chart](misc/loan-default-system-design.png)


```
├── src/
│   ├── jobs/              # ML pipeline components
│   │   ├── train.py       # Model training with hyperparameter tuning
│   │   ├── register.py    # Model registration to MLflow registry
│   │   ├── validate.py    # Data validation and schema checks
│   │   ├── features.py    # Feature engineering pipeline
│   │   ├── data_stats.py  # Dataset statistics computation
│   │   └── drift_check.py # Feature and model drift detection
│   ├── api_layer/         # FastAPI inference service
│   │   ├── main.py        # API endpoints
│   │   └── Dockerfile     # Container configuration
│   ├── config.py          # Centralized configuration
│   └── logging_config.py  # Structured logging setup
├── tests/                 # Unit and integration tests
├── .github/workflows/     # CI/CD automation
└── docker-compose.yaml    # MLflow server orchestration
```

## Quick Start

### Prerequisites
- Python 3.14+
- Docker and Docker Compose
- uv package manager (or pip)

### 1. Start MLflow Server
```bash
docker-compose up -d
```
Access MLflow UI at `http://localhost:5000`

### 2. Train and Register Model
```bash
# Train model with experiment tracking
uv run python src/jobs/train.py --data-path test_data/credit_risk_dataset.csv

# Register best model with @staging alias (F1 score >= 0.80)
uv run python src/jobs/register.py --min-f1-score 0.80
```

The training job logs all parameters, metrics, and artifacts to MLflow. The registration job promotes the best run to the model registry with the `@staging` alias. Manually promote to `@production` in the MLflow UI when ready for deployment.

### 3. Deploy API Service
```bash
uv run uvicorn src.api_layer.main:app --port 8000
```

### 4. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"person_age": 25, "person_income": 50000, "person_emp_length": 3, "loan_amnt": 10000, "loan_int_rate": 8.5, "loan_percent_income": 0.2, "cb_person_cred_hist_length": 5}]}'
```

The API automatically fetches the model tagged with `@production` from the MLflow registry.

## Production Capabilities

### Data Validation and Feature Engineering
The system includes data validation to ensure input quality and feature engineering to prepare data for modeling:

```bash
# Validation runs automatically during training
# Features are engineered via features.py pipeline
```

### Dataset Statistics
Generate comprehensive statistics for exploratory analysis or monitoring:

```bash
uv run python src/jobs/data_stats.py --data-path test_data/credit_risk_dataset.csv --output stats.json
```

### Drift Monitoring

The system implements two-level drift detection for production monitoring:

**Feature Drift** - PSI (Population Stability Index) to detect distribution shifts:
```bash
uv run python src/jobs/drift_check.py --current-data-path test_data/test_data.csv --output drift_report.json
```

PSI thresholds:
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Moderate drift (review recommended)
- PSI ≥ 0.2: Significant drift (action required)

**Model Performance Drift** - Binomial 3-sigma control limits for prediction rates:
- Monitors positive classification rate against baseline
- Upper and Lower Control Limits (UCL/LCL) computed using binomial standard deviation
- Flags when prediction distribution exceeds ±3σ bounds
- Indicate and CI/CD

The project demonstrates professional testing practices with automated continuous integration.

### Test Coverage

**37 automated tests** covering:
- Unit tests for data statistics and drift calculations
- Integration tests for MLflow tracking, model registry, and experiment management
- Tests run in isolation with proper fixtures and mocking

```bash
# Run all tests (automated script)
.\run_tests.ps1  # Windows
./run_tests.sh   # Linux/Mac

# Run specific test types
pytest -m unit -v          # Unit tests only
pytest -m integration -v   # Integration tests (requires MLflow)
```

### Continuous Integration

**GitHub Actions pipeline** (`.github/workflows/ci.yml`):
1. Executes unit tests for core logic
2. Spins up MLflow server via Docker Compose
3. Runs integration tests against live MLflow instance
4. Validates all components work together
5. Reports test results and artifacts

All tests must pass before code can be merged, ensuring production-grade quality.

## Configuration

Centralized configuration management via environment variables and `src/config.py`:

```bash
# Optional: customize settings
cp .env.example .env
```

Key configurations:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MIN_F1_SCORE`: Minimum F1 threshold for production promotion
- `PSI_THRESHOLD_HIGH`: Feature drift alert threshold
- `N_BINS_FOR_NUMERICAL`: Binning strategy for drift calculations

## Technical Highlights

**MLOps Best Practices:**
- Versioned experiments with MLflow tracking server
- Model registry with staging/production promotion workflow
- Automated model registration to @staging based on performance thresholds
- Manual promotion to @production for controlled deployments
- Containerized deployment with Docker

**Monitoring and Observability:**
- Statistical drift detection (PSI)
- Model performance monitoring
- Structured logging throughout the pipeline
- Comprehensive error handling

**Code Quality:**
- Type hints and docstrings
- Modular architecture with separation of concerns
- Centralized configuration management
- 100% test pass rate with CI enforcement

For more details, see [tests/README.md](tests/README.md).
