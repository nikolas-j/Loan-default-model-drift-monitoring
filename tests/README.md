# Test Configuration

## Running Tests Locally

### Unit Tests
Run unit tests that don't require external services:
```bash
pytest tests/test_data_stats.py tests/test_drift_check.py -v
```

### Integration Tests with MLflow
Integration tests require a running MLflow server. Start MLflow first:
```bash
# Start MLflow with docker-compose
docker-compose up -d

# Wait a few seconds for MLflow to initialize
sleep 5

# Run integration tests
MLFLOW_TRACKING_URI=http://localhost:5000 pytest tests/test_mlflow_integration.py -v

# Stop MLflow
docker-compose down -v
```

### Run All Tests
```bash
# Start MLflow
docker-compose up -d
sleep 5

# Run all tests
MLFLOW_TRACKING_URI=http://localhost:5000 pytest tests/ -v

# Cleanup
docker-compose down -v
```

## Test Structure

- **Unit Tests**: Located in `test_data_stats.py` and `test_drift_check.py`
  - Test individual functions without external dependencies
  - Fast execution, no setup required
  
- **Integration Tests**: Located in `test_mlflow_integration.py`
  - Test MLflow connection and functionality
  - Require running MLflow server
  - Test experiment creation, run logging, model registry

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
1. Runs unit tests
2. Starts MLflow via docker-compose
3. Runs integration tests
4. Reports results

The workflow triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

## Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://localhost:5000`)

## Troubleshooting

If integration tests fail:
1. Check MLflow is running: `curl http://localhost:5000/health`
2. View MLflow logs: `docker-compose logs mlflow`
3. Verify network connectivity
4. Check firewall settings (port 5000 must be accessible)
