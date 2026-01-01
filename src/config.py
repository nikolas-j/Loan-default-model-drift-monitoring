"""
Configuration management for the MLflow Credit Risk project.
Uses environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Central config for all project settings."""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "loan-default-model")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-prediction")
    
    # Model Configuration
    MODEL_ALIAS_PRODUCTION: str = "production"
    MODEL_ALIAS_STAGING: str = "staging"
    MIN_F1_SCORE_FOR_PRODUCTION: float = float(os.getenv("MIN_F1_SCORE", "0.85"))
    
    # Data Configuration
    DATA_DIR: Path = PROJECT_ROOT / "test_data"
    BASELINE_DATA_PATH: Path = DATA_DIR / "credit_risk_dataset.csv"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Training Configuration
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    
    # Drift Monitoring Configuration
    PSI_THRESHOLD_LOW: float = 0.1
    PSI_THRESHOLD_HIGH: float = 0.2
    N_BINS_FOR_NUMERICAL: int = int(os.getenv("N_BINS_FOR_NUMERICAL", "10"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_model_uri(cls, alias: Optional[str] = None, version: Optional[int] = None) -> str:
        """
        Get MLflow model URI.
        
        Args:
            alias: Model alias (e.g., 'production', 'staging')
            version: Specific model version number
            
        Returns:
            MLflow model URI string
        """
        if alias:
            return f"models:/{cls.MLFLOW_MODEL_NAME}@{alias}"
        elif version:
            return f"models:/{cls.MLFLOW_MODEL_NAME}/{version}"
        else:
            return f"models:/{cls.MLFLOW_MODEL_NAME}@{cls.MODEL_ALIAS_PRODUCTION}"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if not cls.BASELINE_DATA_PATH.exists():
            raise FileNotFoundError(f"Baseline data not found at {cls.BASELINE_DATA_PATH}")
        
        if cls.MIN_F1_SCORE_FOR_PRODUCTION < 0 or cls.MIN_F1_SCORE_FOR_PRODUCTION > 1:
            raise ValueError("MIN_F1_SCORE_FOR_PRODUCTION must be between 0 and 1")
        
        if cls.TEST_SIZE <= 0 or cls.TEST_SIZE >= 1:
            raise ValueError("TEST_SIZE must be between 0 and 1")


config = Config()
