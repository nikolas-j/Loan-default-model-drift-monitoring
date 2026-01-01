import pytest
import pandas as pd
import numpy as np
import os
from src.jobs.drift_check import (
    bin_continuous_column,
    bin_categorical_column,
    bin_column_pair,
    calc_PSI_for_feature
)


pytestmark = pytest.mark.unit


@pytest.fixture
def baseline_data():
    """Load baseline data from test_data."""
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_data",
        "credit_risk_dataset.csv"
    )
    return pd.read_csv(test_data_path)


@pytest.fixture
def actual_data(baseline_data):
    """Create actual data by sampling 20% of baseline data."""
    return baseline_data.sample(frac=0.2, random_state=42)


class TestBinContinuousColumn:
    """Test suite for bin_continuous_column function."""
    
    def test_basic_binning(self, baseline_data, actual_data):
        """Test that continuous columns are binned based on baseline distribution."""
        baseline_col = baseline_data['person_age']
        actual_col = actual_data['person_age']
        
        baseline_binned, actual_binned = bin_continuous_column(baseline_col, actual_col, n_bins=10)
        
        # Check that both are binned
        assert baseline_binned.notna().sum() > 0
        assert actual_binned.notna().sum() > 0
        
        # Check that bins are within expected range
        assert baseline_binned.max() <= 9  # 10 bins means max bin index is 9
        
    def test_bins_based_on_baseline(self):
        """Test that bins are created from baseline distribution."""
        baseline_col = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        actual_col = pd.Series([5, 15, 25, 35, 45])
        
        baseline_binned, actual_binned = bin_continuous_column(baseline_col, actual_col, n_bins=5)
        
        # Baseline should have values in all bins
        assert len(baseline_binned.dropna()) > 0
        # Actual should be binned using same edges
        assert len(actual_binned.dropna()) > 0
        
    def test_actual_values_outside_baseline_range(self):
        """Test handling when actual data has values outside baseline range."""
        baseline_col = pd.Series([10, 20, 30, 40, 50])
        actual_col = pd.Series([5, 60, 25, 35])  # 5 and 60 are outside baseline range
        
        baseline_binned, actual_binned = bin_continuous_column(baseline_col, actual_col, n_bins=5)
        
        # Should handle out-of-range values (they become NaN)
        assert isinstance(baseline_binned, pd.Series)
        assert isinstance(actual_binned, pd.Series)
        
    def test_same_data_produces_identical_bins(self):
        """Test that identical data produces identical binning."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        baseline_binned, actual_binned = bin_continuous_column(data, data, n_bins=5)
        
        # Should be identical
        pd.testing.assert_series_equal(baseline_binned, actual_binned)


class TestBinCategoricalColumn:
    """Test suite for bin_categorical_column function."""
    
    def test_basic_categorical_binning(self, baseline_data, actual_data):
        """Test that categorical columns are converted to numeric codes."""
        baseline_col = baseline_data['person_home_ownership']
        actual_col = actual_data['person_home_ownership']
        
        baseline_binned, actual_binned = bin_categorical_column(baseline_col, actual_col)
        
        # Check that both are numeric
        assert pd.api.types.is_numeric_dtype(baseline_binned)
        assert pd.api.types.is_numeric_dtype(actual_binned)
        
    def test_same_categories_enforced(self):
        """Test that actual data uses same categories as baseline."""
        baseline_col = pd.Series(['A', 'B', 'C', 'A', 'B'])
        actual_col = pd.Series(['A', 'B', 'A'])  # Missing 'C'
        
        baseline_binned, actual_binned = bin_categorical_column(baseline_col, actual_col)
        
        # Both should have codes for same categories
        assert baseline_binned.max() == 2  # Categories A, B, C
        assert actual_binned.min() >= 0  # Valid codes
        
    def test_new_category_in_actual(self):
        """Test handling when actual has categories not in baseline."""
        baseline_col = pd.Series(['A', 'B', 'C'])
        actual_col = pd.Series(['A', 'B', 'D'])  # 'D' not in baseline
        
        baseline_binned, actual_binned = bin_categorical_column(baseline_col, actual_col)
        
        # 'D' should get code -1 (unknown category)
        assert (actual_binned == -1).any()
        
    def test_empty_actual_category(self):
        """Test when actual is missing a category present in baseline."""
        baseline_col = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'])
        actual_col = pd.Series(['A', 'A'])  # Only 'A', missing 'B' and 'C'
        
        baseline_binned, actual_binned = bin_categorical_column(baseline_col, actual_col)
        
        # Actual should only have code for 'A'
        assert actual_binned.nunique() == 1


class TestBinColumnPair:
    """Test suite for bin_column_pair function."""
    
    def test_numerical_column_dispatch(self, baseline_data, actual_data):
        """Test that numerical columns are handled correctly."""
        baseline_col = baseline_data['person_income']
        actual_col = actual_data['person_income']
        
        baseline_binned, actual_binned = bin_column_pair(baseline_col, actual_col, n_bins=10)
        
        assert pd.api.types.is_numeric_dtype(baseline_binned)
        assert pd.api.types.is_numeric_dtype(actual_binned)
        
    def test_categorical_column_dispatch(self, baseline_data, actual_data):
        """Test that categorical columns are handled correctly."""
        baseline_col = baseline_data['loan_intent']
        actual_col = actual_data['loan_intent']
        
        baseline_binned, actual_binned = bin_column_pair(baseline_col, actual_col, n_bins=10)
        
        assert pd.api.types.is_numeric_dtype(baseline_binned)
        assert pd.api.types.is_numeric_dtype(actual_binned)
        
    def test_processes_all_columns(self, baseline_data, actual_data):
        """Test that function can process all columns in the dataset."""
        for column in baseline_data.columns:
            baseline_binned, actual_binned = bin_column_pair(
                baseline_data[column], 
                actual_data[column], 
                n_bins=10
            )
            
            # Should return Series for both
            assert isinstance(baseline_binned, pd.Series)
            assert isinstance(actual_binned, pd.Series)


class TestCalcPSIForFeature:
    """Test suite for calc_PSI_for_feature function."""
    
    def test_identical_distributions(self):
        """Test PSI calculation for identical distributions."""
        expected = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        actual = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        psi_list = calc_PSI_for_feature(expected, actual)
        total_psi = sum(psi_list)
        
        # PSI should be very close to 0 for identical distributions
        assert total_psi < 0.01
        
    def test_different_distributions(self):
        """Test PSI calculation for different distributions."""
        expected = pd.Series([0] * 50 + [1] * 50)
        actual = pd.Series([0] * 80 + [1] * 20)
        
        psi_list = calc_PSI_for_feature(expected, actual)
        total_psi = sum(psi_list)
        
        # PSI should be positive for different distributions
        assert total_psi > 0.1
        
    def test_returns_list(self):
        """Test that the function returns a list."""
        expected = pd.Series([0, 1, 2, 0, 1, 2])
        actual = pd.Series([0, 1, 2, 0, 1, 2])
        
        psi_list = calc_PSI_for_feature(expected, actual)
        
        assert isinstance(psi_list, list)
        assert len(psi_list) > 0
        
    def test_handles_missing_bins(self):
        """Test handling when actual data has bins not in expected."""
        expected = pd.Series([0, 0, 1, 1])
        actual = pd.Series([0, 1, 2, 2])  # Bin 2 not in expected
        
        psi_list = calc_PSI_for_feature(expected, actual)
        
        # Should not raise error and return valid PSI
        assert isinstance(psi_list, list)
        assert all(isinstance(x, (int, float)) for x in psi_list)
        
    def test_psi_with_real_data(self, baseline_data, actual_data):
        """Test PSI calculation with real credit risk data."""
        # Bin the data first
        baseline_binned, actual_binned = bin_column_pair(
            baseline_data['person_age'], 
            actual_data['person_age'], 
            n_bins=10
        )
        
        psi_list = calc_PSI_for_feature(baseline_binned, actual_binned)
        total_psi = sum(psi_list)
        
        # PSI should be computed successfully
        assert isinstance(total_psi, float)
        assert total_psi >= 0  # PSI is always non-negative
        
    def test_psi_all_features(self, baseline_data, actual_data):
        """Test PSI calculation for all features in dataset."""
        for column in baseline_data.columns:
            baseline_binned, actual_binned = bin_column_pair(
                baseline_data[column], 
                actual_data[column], 
                n_bins=10
            )
            
            psi_list = calc_PSI_for_feature(baseline_binned, actual_binned)
            total_psi = sum(psi_list)
            
            # PSI should be valid for all features
            assert isinstance(total_psi, float)
            assert total_psi >= 0
            assert not np.isnan(total_psi)
