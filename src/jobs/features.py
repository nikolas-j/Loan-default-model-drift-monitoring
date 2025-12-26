import pandas as pd

def data_to_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert raw data into features suitable for model training.

    Args:
        data (pd.DataFrame): Raw input data.

    Returns:
        features, target (DataFrame, Series): Processed feature and target set.
    """
    
    target = data["loan_status"].copy()
    data = data.drop(columns=["loan_status", "loan_grade"])
    clean = data.dropna(axis='index', how="any", inplace=False).drop_duplicates()
    
    # One-hot encode categorical columns
    data_encoded = pd.get_dummies(
        clean, 
        columns=['cb_person_default_on_file', 'person_home_ownership', 'loan_intent'], 
        drop_first=True
    )
    
    bool_cols = data_encoded.select_dtypes(include='bool').columns
    data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)
    
    target = target.loc[data_encoded.index]

    return data_encoded, target