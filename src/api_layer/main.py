from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
import mlflow
import pandas as pd
import os
from src.api_layer.schemas import PredictionRequest, PredictionResponse

# API-specific configuration from environment variables (Docker-friendly)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan-default-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Load model with @production alias
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model loaded successfully: {model_uri}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Make sure model '{MODEL_NAME}' exists with '{MODEL_ALIAS}' alias in MLflow")
        raise
    
    yield
    # Cleanup if needed

app = FastAPI(
    title="Loan Default Prediction API",
    description="ML inference service using MLflow model registry",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }

@app.get("/health")
async def health():
    """Detailed health check including model status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "mlflow_uri": MLFLOW_TRACKING_URI
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on tabular data.
    
    Expects a list of records with feature values matching the model's signature.
    Returns predictions (0 or 1 for loan default).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame(request.data)
        predictions = model.predict(df)
        model_meta = model.metadata
        model_version = model_meta.get_model_info().get("version", "unknown")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_name=MODEL_NAME,
            model_version=model_version
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_meta = model.metadata
        signature = model_meta.signature
        
        return {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "signature": {
                "inputs": str(signature.inputs) if signature else None,
                "outputs": str(signature.outputs) if signature else None
            }
        }
    except Exception as e:
        return {"error": str(e)}

