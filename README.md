# MLFlow Experiment - Loan Default Prediction

Gradient Boosting Classifier to predict loan defaults. Tracks experiments with MLflow, registers production models, and serves predictions via FastAPI.

## Components

**MLflow Server**: Experiment tracking and model registry (SQLite backend, local artifacts)  
**Training Script**: Trains models with hyperparameter tuning, logs metrics/artifacts to MLflow  
**Register Script**: Promotes best model (by F1 score) to registry with @production alias  
**API Layer**: FastAPI service that loads registered model and serves predictions

## Quick Start

```bash
# 1. Start MLflow server
docker-compose up -d

# 2. Train model (run multiple times to compare)
uv run python src/jobs/train.py --data-path test_data/credit_risk_dataset.csv

# 3. Register best model (F1 >= 0.85)
uv run python src/jobs/register.py --min-f1-score 0.85

# 4. Start API server
uv run uvicorn src.api_layer.main:app --port 8000

# 5. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"person_age": 25, "person_income": 50000, "person_emp_length": 3, "loan_amnt": 10000, "loan_int_rate": 8.5, "loan_percent_income": 0.2, "cb_person_cred_hist_length": 5}]}'
```
