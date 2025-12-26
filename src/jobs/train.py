import pandas as pd
import scipy
import mlflow

import argparse
from typing import Dict
from pathlib import Path
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

from typing import Dict

from features import data_to_features
from validate import validate_raw_data

print("Imports successful.")

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Time split produced empty train or test set.")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatch between features and target sizes after split.")

    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier()
    param_dist = {
        "min_samples_split" : scipy.stats.randint(2,10),
        "min_samples_leaf" : scipy.stats.randint(1,10),
        "max_depth" : scipy.stats.randint(3,10)
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=3, cv=2, scoring='f1', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    return metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gradient Boosting Classifier with MLflow tracking.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000", help="MLflow tracking server URI.")
    parser.add_argument("--experiment-name", type=str, default="loan-default-experiment", help="MLflow experiment name.")
    parser.add_argument("--model-name", type=str, default="loan-default-model", help="Registered model name in MLflow.")
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_name = args.experiment_name
    model_name = args.model_name

    # === MLFlow setup === #
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("Starting training pipeline...")

    # === Data === #
    print("Loading data...")
    data = load_data(args.data_path)
    if validate_raw_data(data) == False:
        raise ValueError("Data validation failed.")
    X, y = data_to_features(data)

    # === Time Series Split === #
    print("Splitting data...")
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

    # === Train === #
    params_to_log = {
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "feature_count": X_train.shape[1],
    }

    print("Setting up MLFlow...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(params_to_log)

        print("Training model...")
        model = train_model(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log dataset path as a tag for easy filtering
        mlflow.set_tag("dataset_path", args.data_path)
        
        print("Logging model...")
        input_example = X_train.head(5)
        signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

        # Only log model (not registered - that happens in register.py)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        output = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics,
            "params": params_to_log
        }

    print(output)
    print("Training pipeline ended.")

if __name__ == "__main__":
    main()