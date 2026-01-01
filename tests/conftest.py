"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that require MLflow server"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their module."""
    for item in items:
        # Mark integration tests
        if "test_mlflow_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark unit tests
        elif "test_data_stats" in str(item.fspath) or "test_drift_check" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
