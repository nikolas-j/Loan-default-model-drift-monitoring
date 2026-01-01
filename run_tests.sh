#!/bin/bash
# Script to run all tests (unit + integration) locally

set -e  # Exit on any error

echo "================================"
echo "Running Test Suite"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Run unit tests
echo -e "${BLUE}Step 1: Running unit tests...${NC}"
pytest tests/test_data_stats.py tests/test_drift_check.py -v

echo -e "${GREEN}✓ Unit tests passed!${NC}"
echo ""

# Step 2: Start MLflow
echo -e "${BLUE}Step 2: Starting MLflow with docker-compose...${NC}"
docker-compose up -d

# Step 3: Wait for MLflow
echo -e "${BLUE}Step 3: Waiting for MLflow to be ready...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null 2>&1; then
        echo -e "${GREEN}✓ MLflow is ready!${NC}"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}✗ MLflow failed to start${NC}"
    docker-compose logs mlflow
    docker-compose down -v
    exit 1
fi

echo ""

# Step 4: Run integration tests
echo -e "${BLUE}Step 4: Running integration tests...${NC}"
export MLFLOW_TRACKING_URI=http://localhost:5000
pytest tests/test_mlflow_integration.py -v

echo -e "${GREEN}✓ Integration tests passed!${NC}"
echo ""

# Step 5: Cleanup
echo -e "${BLUE}Step 5: Cleaning up...${NC}"
docker-compose down -v

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}All tests passed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
