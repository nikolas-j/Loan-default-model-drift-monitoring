"""
Integration tests for MLflow tracking and model registry.
These tests require a running MLflow server (via docker-compose).
"""
import pytest
import mlflow
import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mlflow_client():
    """Setup MLflow client with tracking URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Wait for MLflow to be ready (with retries)
    max_retries = 30
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            # Test connection by listing experiments
            client.search_experiments()
            print(f"✓ Connected to MLflow at {tracking_uri}")
            return client
        except Exception as e:
            if i < max_retries - 1:
                print(f"Waiting for MLflow... ({i+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                pytest.fail(f"Could not connect to MLflow after {max_retries} attempts: {e}")


@pytest.fixture(scope="module")
def test_experiment_name():
    """Generate unique experiment name for testing."""
    return f"integration-test-{int(time.time())}"


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


class TestMLflowConnection:
    """Test MLflow server connectivity and basic operations."""
    
    def test_mlflow_server_is_running(self, mlflow_client):
        """Verify MLflow server is accessible and responding."""
        assert mlflow_client is not None
        
    def test_can_list_experiments(self, mlflow_client):
        """Verify we can list experiments from MLflow."""
        experiments = mlflow_client.search_experiments()
        assert isinstance(experiments, list)
        
    def test_tracking_uri_is_set(self):
        """Verify MLflow tracking URI is configured."""
        tracking_uri = mlflow.get_tracking_uri()
        assert tracking_uri is not None
        assert "localhost:5000" in tracking_uri or "127.0.0.1:5000" in tracking_uri


class TestExperimentCreation:
    """Test experiment creation and management."""
    
    def test_create_experiment(self, mlflow_client, test_experiment_name):
        """Test creating a new experiment."""
        experiment_id = mlflow.create_experiment(test_experiment_name)
        assert experiment_id is not None
        
        # Verify experiment was created
        experiment = mlflow_client.get_experiment(experiment_id)
        assert experiment.name == test_experiment_name
        
    def test_set_experiment(self, test_experiment_name):
        """Test setting an active experiment."""
        mlflow.set_experiment(test_experiment_name)
        
        # Verify experiment is active
        experiment = mlflow.get_experiment_by_name(test_experiment_name)
        assert experiment is not None
        assert experiment.name == test_experiment_name


class TestRunLogging:
    """Test MLflow run creation and logging capabilities."""
    
    def test_create_run(self, test_experiment_name):
        """Test creating a new MLflow run."""
        mlflow.set_experiment(test_experiment_name)
        
        with mlflow.start_run() as run:
            assert run.info.run_id is not None
            assert run.info.experiment_id is not None
            
    def test_log_parameters(self, test_experiment_name, mlflow_client):
        """Test logging parameters to a run."""
        mlflow.set_experiment(test_experiment_name)
        
        params = {
            "learning_rate": 0.01,
            "n_estimators": 100,
            "max_depth": 5
        }
        
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            
        # Verify parameters were logged
        run_data = mlflow_client.get_run(run.info.run_id)
        for key, value in params.items():
            assert key in run_data.data.params
            assert run_data.data.params[key] == str(value)
            
    def test_log_metrics(self, test_experiment_name, mlflow_client):
        """Test logging metrics to a run."""
        mlflow.set_experiment(test_experiment_name)
        
        metrics = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "precision": 0.93,
            "recall": 0.91
        }
        
        with mlflow.start_run() as run:
            mlflow.log_metrics(metrics)
            
        # Verify metrics were logged
        run_data = mlflow_client.get_run(run.info.run_id)
        for key, value in metrics.items():
            assert key in run_data.data.metrics
            assert abs(run_data.data.metrics[key] - value) < 1e-6
            
    def test_log_tags(self, test_experiment_name, mlflow_client):
        """Test logging tags to a run."""
        mlflow.set_experiment(test_experiment_name)
        
        tags = {
            "model_type": "gradient_boosting",
            "dataset": "credit_risk",
            "environment": "test"
        }
        
        with mlflow.start_run() as run:
            mlflow.set_tags(tags)
            
        # Verify tags were logged
        run_data = mlflow_client.get_run(run.info.run_id)
        for key, value in tags.items():
            assert key in run_data.data.tags
            assert run_data.data.tags[key] == value


class TestModelLogging:
    """Test model logging and retrieval."""
    
    def test_log_sklearn_model(self, test_experiment_name, sample_data):
        """Test logging a scikit-learn model."""
        mlflow.set_experiment(test_experiment_name)
        X, y = sample_data
        
        # Train a simple model
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        
        with mlflow.start_run() as run:
            # Log model with signature
            signature = mlflow.models.infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X.head(3)
            )
            
        # Verify model was logged
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        assert loaded_model is not None
        
        # Verify model can make predictions
        predictions = loaded_model.predict(X.head(5))
        assert len(predictions) == 5
        
    def test_model_signature(self, test_experiment_name, sample_data):
        """Test that model signature is properly saved."""
        mlflow.set_experiment(test_experiment_name)
        X, y = sample_data
        
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        with mlflow.start_run() as run:
            signature = mlflow.models.infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(
                sk_model=model,
                registered_model_name="model",
                signature=signature
            )
            
        # Load and verify signature
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        assert loaded_model.metadata.signature is not None


class TestModelRegistry:
    """Test MLflow Model Registry functionality."""
    
    def test_register_model(self, test_experiment_name, sample_data, mlflow_client):
        """Test registering a model to the model registry."""
        mlflow.set_experiment(test_experiment_name)
        X, y = sample_data
        
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        model_name = f"test-model-{int(time.time())}"
        
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )
            
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        assert registered_model.name == model_name
        assert registered_model.version is not None
        
        # Cleanup: delete registered model
        try:
            mlflow_client.delete_registered_model(model_name)
        except:
            pass  # Ignore cleanup errors
            
    def test_load_registered_model(self, test_experiment_name, sample_data, mlflow_client):
        """Test loading a model from the registry."""
        mlflow.set_experiment(test_experiment_name)
        X, y = sample_data
        
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        model_name = f"test-model-load-{int(time.time())}"
        
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )
            
        # Register and load the model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Load from registry
        loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/{registered_model.version}")
        assert loaded_model is not None
        
        # Test predictions
        predictions = loaded_model.predict(X.head(5))
        assert len(predictions) == 5
        
        # Cleanup
        try:
            mlflow_client.delete_registered_model(model_name)
        except:
            pass


class TestSearchRuns:
    """Test searching and filtering runs."""
    
    def test_search_runs_by_metrics(self):
        """Test searching runs by metric values."""
        # Use unique experiment for this test
        exp_name = f"test-search-metrics-{int(time.time() * 1000)}"
        mlflow.set_experiment(exp_name)
        
        # Create multiple runs with different metrics
        f1_scores = [0.85, 0.90, 0.88]
        for score in f1_scores:
            with mlflow.start_run():
                mlflow.log_metric("f1_score", score)
                mlflow.log_param("model_type", "test")
                
        # Search for runs with f1_score > 0.87
        runs = mlflow.search_runs(
            experiment_names=[exp_name],
            filter_string="metrics.f1_score > 0.87",
            order_by=["metrics.f1_score DESC"]
        )
        
        assert len(runs) == 2  # 0.90 and 0.88
        assert runs.iloc[0]["metrics.f1_score"] == 0.90
        
    def test_search_best_run(self):
        """Test finding the best run by a metric."""
        # Use unique experiment for this test
        exp_name = f"test-search-best-{int(time.time() * 1000)}"
        mlflow.set_experiment(exp_name)
        
        # Create runs with different scores
        scores = [0.75, 0.92, 0.88, 0.85]
        for score in scores:
            with mlflow.start_run():
                mlflow.log_metric("accuracy", score)
                
        # Find best run
        runs = mlflow.search_runs(
            experiment_names=[exp_name],
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )
        
        assert len(runs) == 1
        assert runs.iloc[0]["metrics.accuracy"] == 0.92
