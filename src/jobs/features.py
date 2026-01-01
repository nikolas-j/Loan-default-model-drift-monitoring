import pandas as pd
from src.logging_config import setup_logging

logger = setup_logging(__name__)

def data_to_features(data: pd.DataFrame, training: bool=True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert raw data into features suitable for model training.

    Args:
        data (pd.DataFrame): Raw input data.
        training (bool): Flag indicating if the data is for training (True) or inference (False).
    Returns:
        features, target (DataFrame, Series): Processed feature and target set.
    """
    
    if training:
        target = data["loan_status"].copy()
    else:
        target = None
    data = data.drop(columns=["loan_status", "loan_grade"])

    # NOTE: Choose here what to do with missing values and duplicates. GBClassifier cannot handle natively Nans.
    original_size = len(data)
    data = data.dropna(axis='index', how="any", inplace=False).drop_duplicates()
    rows_removed = original_size - len(data)
    if rows_removed > 0:
        logger.info(f"Removed {rows_removed} rows (NaNs/duplicates) from {original_size} total")
    
    # One-hot encode categorical columns
    data_encoded = pd.get_dummies(
        data, 
        columns=['cb_person_default_on_file', 'person_home_ownership', 'loan_intent'], 
        drop_first=True # Reduce multicollinearity
    )
    
    bool_cols = data_encoded.select_dtypes(include='bool').columns
    data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)
    
    if training:
        target = target.loc[data_encoded.index]

    return data_encoded, target