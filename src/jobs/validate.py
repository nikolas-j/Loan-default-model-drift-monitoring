import pandas as pd
from src.logging_config import setup_logging

logger = setup_logging(__name__)

def validate_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw input data for correct columns and types.

    Args:
        data (pd.DataFrame): Raw input data.

    Returns:
        True if data is valid, raises ValueError otherwise.
    """

    if data.empty:
        logger.error("Validation failed: Empty dataframe")
        return False
    
    required_columns = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]

    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Validation failed: Missing required column '{col}'")
            return False
    
    logger.info("Data validation passed")
    return True