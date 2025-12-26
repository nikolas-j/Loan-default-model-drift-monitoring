import mlflow
import argparse
from typing import Optional


def find_best_run(experiment_name: str) -> Optional[dict]:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        print(f"No runs found in experiment '{experiment_name}'")
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
    """Register model if it meets the F1 score threshold."""
    
    if f1_score < min_f1_threshold:
        print(f"Model F1 score {f1_score:.4f} below threshold {min_f1_threshold:.4f}")
        print("Model not registered.")
        return None
    
    model_uri = f"runs:/{run_id}/model"
    
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Registered as version {result.version}")
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

    # Fancy output    
    print("=" * 60)
    print("MLflow Model Registration")
    print("=" * 60)
    print(f"Experiment: {args.mlflow_experiment_name}")
    print(f"Model name: {args.model_name}")
    print(f"Min F1 threshold: {args.min_f1_score:.4f}")
    print()
    
    print("Searching for best run by F1 score...")
    best_run = find_best_run(args.mlflow_experiment_name)
    
    if best_run is None:
        return
    
    # Display best run metrics
    print()
    print("Best run found:")
    print(f"  Run ID: {best_run['run_id']}")
    print(f"  F1 Score: {best_run['f1_score']:.4f}")
    print(f"  Accuracy: {best_run['accuracy']:.4f}")
    print(f"  Precision: {best_run['precision']:.4f}")
    print(f"  Recall: {best_run['recall']:.4f}")
    print()
    
    version = register_model(
        run_id=best_run["run_id"],
        model_name=args.model_name,
        f1_score=best_run["f1_score"],
        min_f1_threshold=args.min_f1_score
    )
    
    if version:
        print()
        print("=" * 60)
        print(f"Model registered as '{args.model_name}' v{version}")
        print("=" * 60)


if __name__ == "__main__":
    main()
