import pandas as pd
import numpy as np
from src.config import config
from src.logging_config import setup_logging, log_exception

logger = setup_logging(__name__)
from src.config import config

def bin_continuous_column(baseline_col: pd.Series, actual_col: pd.Series, 
                         n_bins: int = 10) -> tuple[pd.Series, pd.Series]:

    # Create bin edges from baseline distribution
    bins = np.linspace(baseline_col.min(), baseline_col.max(), n_bins + 1)
    
    baseline_binned = pd.cut(baseline_col, bins=bins, labels=False, include_lowest=True)
    actual_binned = pd.cut(actual_col, bins=bins, labels=False, include_lowest=True)
    
    return baseline_binned, actual_binned


def bin_categorical_column(baseline_col: pd.Series, actual_col: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Convert baseline to categorical and get the categories
    baseline_cat = baseline_col.astype('category')
    baseline_categories = baseline_cat.cat.categories
    
    # Convert actual to categorical with the same categories as baseline
    actual_cat = actual_col.astype('category')
    actual_cat = actual_cat.cat.set_categories(baseline_categories)
    
    baseline_binned = baseline_cat.cat.codes
    actual_binned = actual_cat.cat.codes
    
    return baseline_binned, actual_binned


def bin_column_pair(baseline_col: pd.Series, actual_col: pd.Series, 
                   n_bins: int = 10) -> tuple[pd.Series, pd.Series]:
    """
    Bin a pair of baseline and actual columns based on baseline distribution.

    Args:
        baseline_col (pd.Series): Baseline feature column.
        actual_col (pd.Series): Actual feature column.
        n_bins (int): Number of bins for numerical features.

    Returns:
        baseline_binned (pd.Series): Binned baseline column.
        actual_binned (pd.Series): Binned actual column.
    """
    # Check if column is numerical or categorical
    if pd.api.types.is_numeric_dtype(baseline_col):
        return bin_continuous_column(baseline_col, actual_col, n_bins)
    else:
        return bin_categorical_column(baseline_col, actual_col)


def calc_PSI_for_feature(expected, actual):
    """
    Calculate Population Stability Index (PSI) for a single feature.

    Args:
        expected (pd.Series): Baseline data for the feature transformed into bins.
        actual (pd.Series): Current data for the feature transformed into bins.
        n_bins_for_numerical (int): Number of bins for numerical features.
    
    Returns:
        psi (list): List of PSI values for each bin.
    """
    
    # Get bin counts for expected and actual data
    expected_counts = expected.value_counts().sort_index()
    actual_counts = actual.value_counts().sort_index()
    
    # Ensure both have the same bins
    all_bins = expected_counts.index.union(actual_counts.index)
    expected_counts = expected_counts.reindex(all_bins, fill_value=0)
    actual_counts = actual_counts.reindex(all_bins, fill_value=0)
    
    # Calculate percentages (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    expected_pct = (expected_counts / expected_counts.sum()) + epsilon
    actual_pct = (actual_counts / actual_counts.sum()) + epsilon
    
    # Calculate PSI for each bin: (actual% - expected%) * ln(actual% / expected%)
    psi_list = []
    for i in range(len(expected_pct)):
        psi_value = (actual_pct.iloc[i] - expected_pct.iloc[i]) * np.log(actual_pct.iloc[i] / expected_pct.iloc[i])
        psi_list.append(psi_value)
    
    return psi_list


def run_drift_check(current_data_path: str):
    """
    Compares baseline training data with current data for feature distribution shift with PSI.
    Runs inference with model alias "@production" on both datasets to report if model performance has drifted.

    Args:
        current_data_path (str): Path to the current data CSV file.

    Returns:
        drift_report (dict): A report with PSI (population stability index) values 
        for each feature and model performance metrics compared with baseline.
    """
    from src.jobs.features import data_to_features
    
    logger.info("="*60)
    logger.info("Starting drift check")
    logger.info("="*60)
    
    # Load baseline data from config
    logger.info(f"Loading baseline data from: {config.BASELINE_DATA_PATH}")
    expected_data = pd.read_csv(config.BASELINE_DATA_PATH)
    logger.info(f"Baseline data loaded: {expected_data.shape[0]} rows, {expected_data.shape[1]} columns")
    
    logger.info(f"Loading current data from: {current_data_path}")
    actual_data = pd.read_csv(current_data_path)
    logger.info(f"Current data loaded: {actual_data.shape[0]} rows, {actual_data.shape[1]} columns")
    
    # Calculate PSI for each feature by binning column pairs
    logger.info(f"Calculating PSI for {len(expected_data.columns)} features...")
    feature_psi = {}
    for column in expected_data.columns:
        expected_binned, actual_binned = bin_column_pair(
            expected_data[column], 
            actual_data[column], 
            n_bins=config.N_BINS_FOR_NUMERICAL
        )
        
        psi_per_bin = calc_PSI_for_feature(expected_binned, actual_binned)
        total_psi = sum(psi_per_bin)
        feature_psi[column] = total_psi
        
        # Log PSI level
        if total_psi < config.PSI_THRESHOLD_LOW:
            logger.info(f"  [OK] {column}: PSI={total_psi:.4f} (No significant drift)")
        elif total_psi < config.PSI_THRESHOLD_HIGH:
            logger.warning(f"  [WARNING] {column}: PSI={total_psi:.4f} (Moderate drift detected)")
        else:
            logger.error(f"  [ALERT] {column}: PSI={total_psi:.4f} (Significant drift detected!)")
    
    logger.info("Preparing data for model inference...")
    X_expected, y_expected = data_to_features(expected_data, training=True)
    X_actual, _ = data_to_features(actual_data, training=False)
    logger.info(f"Data prepared: {X_expected.shape[1]} features")
    
    model_performance_drift = {}
    try:
        import mlflow
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Load the production model
        logger.info(f"Connecting to MLflow at: {config.MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        
        model_uri = config.get_model_uri(alias=config.MODEL_ALIAS_PRODUCTION)
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("[OK] Model loaded successfully")
        
    except Exception as e:
        log_exception(logger, e, "Failed to load model")
        model_performance_drift = {"error": f"Failed to load model: {str(e)}"}
        drift_report = {
            "feature_drift": feature_psi,
            "model_performance_drift": model_performance_drift
        }
        return drift_report
    
    logger.info("Running inference on baseline data...")
    y_pred_expected = model.predict(X_expected)
    logger.info("Running inference on current data...")
    y_pred_actual = model.predict(X_actual)
    
    logger.info("Calculating performance metrics...")
    baseline_metrics = {
        "accuracy": accuracy_score(y_expected, y_pred_expected),
        "precision": precision_score(y_expected, y_pred_expected, zero_division=0),
        "recall": recall_score(y_expected, y_pred_expected, zero_division=0),
        "f1_score": f1_score(y_expected, y_pred_expected, zero_division=0),
        "roc_auc": roc_auc_score(y_expected, y_pred_expected)
    }
    
    logger.info(f"Baseline metrics: F1={baseline_metrics['f1_score']:.4f}, ROC-AUC={baseline_metrics['roc_auc']:.4f}")
    
    # Positive prediction rates
    baseline_positive_rate = float((y_pred_expected == 1).sum() / len(y_pred_expected))
    current_positive_rate = float((y_pred_actual == 1).sum() / len(y_pred_actual))
    
    positive_rate_drift = current_positive_rate - baseline_positive_rate
    
    logger.info(f"Baseline positive prediction rate: {baseline_positive_rate:.4f}")
    logger.info(f"Current positive prediction rate: {current_positive_rate:.4f}")
    logger.info(f"Drift in positive predictions: {positive_rate_drift:+.4f}")
    
    # Calculate 3-sigma control limits for baseline positive rate
    k = 3
    expected_std = np.sqrt(baseline_positive_rate * (1 - baseline_positive_rate) / len(y_expected))
    UCL = baseline_positive_rate + k * expected_std
    LCL = baseline_positive_rate - k * expected_std
    over_confidence_limits = not (LCL <= current_positive_rate <= UCL)
    
    if over_confidence_limits:
        logger.error(f"[ALERT] Prediction rate outside control limits! [{LCL:.4f}, {UCL:.4f}]")
    else:
        logger.info(f"[OK] Prediction rate within control limits [{LCL:.4f}, {UCL:.4f}]")

    # Create model performance drift report
    model_performance_drift = {
        "baseline_metrics": baseline_metrics,
        "baseline_positive_prediction_rate": baseline_positive_rate,
        "current_positive_prediction_rate": current_positive_rate,
        "positive_prediction_rate_drift": positive_rate_drift,
        "UCL": UCL,
        "LCL": LCL,
        "over_confidence_limits": over_confidence_limits
    }
    
    drift_report = {
        "feature_drift": feature_psi,
        "model_performance_drift": model_performance_drift
    }
    
    logger.info("="*60)
    logger.info("Drift check complete")
    logger.info("="*60)
    
    return drift_report


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Check for data and model drift.")
    parser.add_argument("--current-data-path", type=str, required=True, 
                       help="Path to the current data CSV file")
    parser.add_argument("--output", type=str, default=None, 
                       help="Optional path to save drift report as JSON")
    
    args = parser.parse_args()
    
    drift_report = run_drift_check(args.current_data_path)
    
    # Print JSON to console
    print(json.dumps(drift_report, indent=2))
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(drift_report, f, indent=2)
        logger.info(f"Drift report saved to {args.output}")

