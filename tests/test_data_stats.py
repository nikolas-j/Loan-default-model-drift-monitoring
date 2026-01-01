import pytest
import pandas as pd
import numpy as np
from src.jobs.data_stats import run_statistics


pytestmark = pytest.mark.unit


class TestRunStatistics:

    def test_basic_statistics_computation(self):
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000],
            'category': ['A', 'B', 'A', 'C']
        })
        
        stats = run_statistics(data)
        
        assert 'num_rows' in stats
        assert 'num_columns' in stats
        assert 'missing_values' in stats
        assert 'column_types' in stats
        assert 'descriptive_stats' in stats
        
        assert stats['num_rows'] == 4
        assert stats['num_columns'] == 3
        
    def test_missing_values_detection(self):
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [np.nan, np.nan, 3, 4],
            'col3': [1, 2, 3, 4]
        })
        
        stats = run_statistics(data)
        
        assert stats['missing_values']['col1'] == 1
        assert stats['missing_values']['col2'] == 2
        assert stats['missing_values']['col3'] == 0
        
    def test_empty_dataframe(self):
        data = pd.DataFrame()
        
        stats = run_statistics(data)
        
        assert stats['num_rows'] == 0
        assert stats['num_columns'] == 0
        
    def test_column_types_detection(self):
        data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        
        stats = run_statistics(data)
        
        assert 'int' in stats['column_types']['int_col']
        assert 'float' in stats['column_types']['float_col']
        assert 'object' in stats['column_types']['str_col']
        
    def test_descriptive_stats_computed(self):
        data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5]
        })
        
        stats = run_statistics(data)
        
        assert 'values' in stats['descriptive_stats']
        assert 'mean' in stats['descriptive_stats']['values']
        assert 'std' in stats['descriptive_stats']['values']
        assert stats['descriptive_stats']['values']['mean'] == 3.0

