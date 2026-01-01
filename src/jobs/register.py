import mlflow
import argparse
from typing import Optional
from src.logging_config import setup_logging

logger = setup_logging(__name__)

def find_best_run(experiment_name: str) -> Optional[dict]:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        logger.warning(f"No runs found in experiment '{experiment_name}'")
        return None
    
    best_run = runs.iloc[0]
    return {
        "run_id": best_run["run_id"],
        "f1_score": best_run["metrics.f1_score"],
        "accuracy": best_run["metrics.accuracy"],
        "precision": best_run["metrics.precision"],
        "recall": best_run["metrics.recall"],
    }

def register_model(
    run_id: str,
    model_name: str,
    f1_score: float,
    min_f1_threshold: float
) -> Optional[str]:
    """Register model if it meets the F1 score threshold and assign @staging alias."""
    
    if f1_score < min_f1_threshold:
        logger.warning(f"Model F1 score {f1_score:.4f} below threshold {min_f1_threshold:.4f}")
        logger.warning("Model not registered")
        return None
    
    model_uri = f"runs:/{run_id}/model"
    
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Set @staging alias for the newly registered model
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=result.version
    )
    
    logger.info(f"Model registered as '{model_name}' version {result.version}")
    logger.info(f"Assigned alias '@staging' to version {result.version}")
    logger.info("To promote to production, manually set @production alias in MLflow UI")
    return result.version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register best model from MLflow experiment if it meets F1 threshold."
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI."
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="loan-default-experiment",
        help="MLflow experiment name to search for best model."
    )
    parser.add_argument(
        "--model-name",
        default="loan-default-model",
        help="Name for the registered model in MLflow Model Registry."
    )
    parser.add_argument(
        "--min-f1-score",
        type=float,
        default=0.80,
        help="Minimum F1 score threshold for model registration."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    logger.info("=" * 60)
    logger.info("MLflow Model Registration")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.mlflow_experiment_name}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Min F1 threshold: {args.min_f1_score:.4f}")
    
    logger.info("Searching for best run by F1 score...")
    best_run = find_best_run(args.mlflow_experiment_name)
    
    if best_run is None:
        return
    
    # Display best run metrics
    logger.info("Best run found:")
    logger.info(f"  Run ID: {best_run['run_id']}")
    logger.info(f"  F1 Score: {best_run['f1_score']:.4f}")
    logger.info(f"  Accuracy: {best_run['accuracy']:.4f}")
    logger.info(f"  Precision: {best_run['precision']:.4f}")
    logger.info(f"  Recall: {best_run['recall']:.4f}")
    
    version = register_model(
        run_id=best_run["run_id"],
        model_name=args.model_name,
        f1_score=best_run["f1_score"],
        min_f1_threshold=args.min_f1_score
    )
    
    if version:
        logger.info("=" * 60)
        logger.info(f"Model registered as '{args.model_name}' v{version} with @staging alias")
        logger.info("Manually promote to @production in MLflow UI when ready")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
