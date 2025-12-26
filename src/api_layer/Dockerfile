# Production Dockerfile for FastAPI Inference Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (production only, no dev dependencies)
RUN uv pip install --system --no-cache \
    mlflow>=3.8.0 \
    pandas>=2.3.3 \
    scikit-learn>=1.8.0 \
    scipy>=1.16.3 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.34.0

# Copy source code
COPY src/__init__.py src/__init__.py
COPY src/api_layer/ src/api_layer/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose API port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=loan-default-model
ENV MODEL_ALIAS=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the FastAPI application
CMD ["uvicorn", "src.api_layer.main:app", "--host", "0.0.0.0", "--port", "8000"]
