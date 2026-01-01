# PowerShell script to run all tests (unit + integration) locally

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Running Test Suite" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Run unit tests
Write-Host "Step 1: Running unit tests..." -ForegroundColor Blue
pytest tests/test_data_stats.py tests/test_drift_check.py -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Unit tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Unit tests passed!" -ForegroundColor Green
Write-Host ""

# Step 2: Start MLflow
Write-Host "Step 2: Starting MLflow with docker-compose..." -ForegroundColor Blue
docker-compose up -d

# Step 3: Wait for MLflow
Write-Host "Step 3: Waiting for MLflow to be ready..." -ForegroundColor Blue
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/api/2.0/mlflow/experiments/list" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ MLflow is ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }
    
    $attempt++
    Write-Host "Waiting... ($attempt/$maxAttempts)"
    Start-Sleep -Seconds 2
}

if ($attempt -eq $maxAttempts) {
    Write-Host "✗ MLflow failed to start" -ForegroundColor Red
    docker-compose logs mlflow
    docker-compose down -v
    exit 1
}

Write-Host ""

# Step 4: Run integration tests
Write-Host "Step 4: Running integration tests..." -ForegroundColor Blue
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
pytest tests/test_mlflow_integration.py -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Integration tests failed!" -ForegroundColor Red
    docker-compose down -v
    exit 1
}

Write-Host "✓ Integration tests passed!" -ForegroundColor Green
Write-Host ""

# Step 5: Cleanup
Write-Host "Step 5: Cleaning up..." -ForegroundColor Blue
docker-compose down -v

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "All tests passed successfully!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
