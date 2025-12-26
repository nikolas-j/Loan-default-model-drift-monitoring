from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[int]
    model_name: str
    model_version: str