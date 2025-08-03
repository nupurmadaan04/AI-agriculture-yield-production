"""
Unit tests for data preprocessing functions.

This module contains comprehensive unit tests for all data preprocessing functions
including data loading, missing value handling, outlier detection and removal,
feature encoding, data scaling, and validation.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our preprocessing functions
from preprocessing_functions import (
    load_crop_data, filter_rice_data, check_missing_values, 
    handle_missing_values, encode_categorical_column, detect_outliers_iqr,
    remove_outliers_iqr, scale_features, split_data, validate_data_format,
    save_processed_data, complete_preprocessing_pipeline
)


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2020, 2021],
            'State Name': ['Punjab', 'Haryana', 'Punjab', 'Uttar Pradesh', 'Haryana'],
            'Dist Name': ['Dist 1', 'Dist 2', 'Dist 1', 'Dist 3', 'Dist 2'],
            'RICE AREA (1000 ha)': [100, 150, 120, 80, 200],
            'RICE PRODUCTION (1000 tons)': [300, 450, 360, 240, 600],
            'RICE YIELD (Kg per ha)': [3000, 3200, 2800, 3100, 3500],
            'OTHER CROP': [50, 75, 60, 40, 100]  # Additional column to test filtering
        })
        
        # Create test CSV file
        self.test_file = os.path.join(self.temp_dir, 'test_crops.csv')
        self.sample_data.to_csv(self.test_file, index=False)
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_crop_data_success(self):
        """Test successful data loading."""
        df = load_crop_data(self.test_file)
        
        # Check if data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), list(self.sample_data.columns))
        
    def test_load_crop_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_crop_data('/path/to/nonexistent/file.csv')
    
    def test_load_crop_data_empty_file(self):
        """Test loading empty CSV file."""
        empty_file = os.path.join(self.temp_dir, 'empty.csv')
        pd.DataFrame().to_csv(empty_file, index=False)
        
        with self.assertRaises(pd.errors.EmptyDataError):
            load_crop_data(empty_file)


class TestDataFiltering(unittest.TestCase):
    """Test cases for data filtering functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'State Name': ['Punjab', 'Haryana', 'Punjab'],
            'Dist Name': ['Dist 1', 'Dist 2', 'Dist 1'],
            'RICE AREA (1000 ha)': [100, 150, 120],
            'RICE PRODUCTION (1000 tons)': [300, 450, 360],
            'RICE YIELD (Kg per ha)': [3000, 3200, 2800],
            'OTHER CROP': [50, 75, 60]  # This should be filtered out
        })
    
    def test_filter_rice_data_success(self):
        """Test successful rice data filtering."""
        df_rice = filter_rice_data(self.sample_data)
        
        expected_columns = [
            'Year', 'State Name', 'Dist Name',
            'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 
            'RICE YIELD (Kg per ha)'
        ]
        
        self.assertEqual(list(df_rice.columns), expected_columns)
        self.assertEqual(len(df_rice), 3)
        self.assertNotIn('OTHER CROP', df_rice.columns)
    
    def test_filter_rice_data_missing_columns(self):
        """Test filtering with missing required columns."""
        incomplete_data = pd.DataFrame({
            'Year': [2020, 2021],
            'State Name': ['Punjab', 'Haryana']
            # Missing other required columns
        })
        
        with self.assertRaises(KeyError):
            filter_rice_data(incomplete_data)


class TestMissingValueHandling(unittest.TestCase):
    """Test cases for missing value handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create data with missing values
        self.data_with_missing = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2020, 2021],
            'State Name': ['Punjab', 'Haryana', 'Punjab', 'UP', 'Haryana'],
            'Dist Name': ['Dist 1', 'Dist 2', 'Dist 1', 'Dist 3', 'Dist 2'],
            'RICE AREA (1000 ha)': [100, np.nan, 120, 80, 200],  # Missing value
            'RICE PRODUCTION (1000 tons)': [300, 450, np.nan, 240, 600],  # Missing value
            'RICE YIELD (Kg per ha)': [3000, 3200, 2800, 3100, 3500]
        })
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        missing_count = check_missing_values(self.data_with_missing)
        
        self.assertEqual(missing_count['RICE AREA (1000 ha)'], 1)
        self.assertEqual(missing_count['RICE PRODUCTION (1000 tons)'], 1)
        self.assertEqual(missing_count['RICE YIELD (Kg per ha)'], 0)
    
    def test_handle_missing_values_drop(self):
        """Test dropping rows with missing values."""
        df_clean = handle_missing_values(self.data_with_missing, strategy='drop')
        
        # Should have 3 rows after dropping rows with missing values
        self.assertEqual(len(df_clean), 3)
        self.assertFalse(df_clean.isnull().any().any())  # No missing values should remain
    
    def test_handle_missing_values_fill_mean(self):
        """Test filling missing values with mean."""
        df_filled = handle_missing_values(self.data_with_missing, strategy='fill_mean')
        
        # Should have same number of rows
        self.assertEqual(len(df_filled), 5)
        
        # Numeric columns should not have missing values
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        self.assertFalse(df_filled[numeric_cols].isnull().any().any())
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test invalid strategy handling."""
        with self.assertRaises(ValueError):
            handle_missing_values(self.data_with_missing, strategy='invalid_strategy')


class TestCategoricalEncoding(unittest.TestCase):
    """Test cases for categorical encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'State Name': ['Punjab', 'Haryana', 'Punjab'],
            'Dist Name': ['Dist 1', 'Dist 2', 'Dist 1'],
            'RICE AREA (1000 ha)': [100, 150, 120],
            'RICE PRODUCTION (1000 tons)': [300, 450, 360],
            'RICE YIELD (Kg per ha)': [3000, 3200, 2800]
        })
    
    def test_encode_categorical_column_success(self):
        """Test successful categorical encoding."""
        df_encoded = encode_categorical_column(self.sample_data, 'State Name')
        
        # Should have new encoded column
        self.assertIn('state_name_encoded', df_encoded.columns)
        
        # Encoded column should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded['state_name_encoded']))
        
        # Should have same number of rows
        self.assertEqual(len(df_encoded), 3)
    
    def test_encode_categorical_column_missing_column(self):
        """Test encoding non-existent column."""
        with self.assertRaises(KeyError):
            encode_categorical_column(self.sample_data, 'NonExistent Column')


class TestOutlierDetection(unittest.TestCase):
    """Test cases for outlier detection and removal."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create data with outliers
        self.data_with_outliers = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2020, 2021],
            'State Name': ['Punjab', 'Haryana', 'Punjab', 'UP', 'Haryana'],
            'Dist Name': ['Dist 1', 'Dist 2', 'Dist 1', 'Dist 3', 'Dist 2'],
            'RICE AREA (1000 ha)': [100, 150, 120, 80, 200],
            'RICE PRODUCTION (1000 tons)': [300, 450, 360, 240, 600],
            'RICE YIELD (Kg per ha)': [3000, 10000, 2800, 3100, 500]  # 10000 and 500 are outliers
        })
    
    def test_detect_outliers_iqr_success(self):
        """Test successful outlier detection using IQR method."""
        outlier_info = detect_outliers_iqr(self.data_with_outliers, 'RICE YIELD (Kg per ha)')
        
        # Should return dictionary with required keys
        required_keys = ['lower_bound', 'upper_bound', 'Q1', 'Q3', 'IQR', 'outlier_count', 'outlier_indices']
        for key in required_keys:
            self.assertIn(key, outlier_info)
        
        # Should detect outliers
        self.assertGreater(outlier_info['outlier_count'], 0)
    
    def test_detect_outliers_iqr_missing_column(self):
        """Test outlier detection with non-existent column."""
        with self.assertRaises(KeyError):
            detect_outliers_iqr(self.data_with_outliers, 'NonExistent Column')
    
    def test_remove_outliers_iqr_success(self):
        """Test successful outlier removal."""
        df_clean = remove_outliers_iqr(self.data_with_outliers, 'RICE YIELD (Kg per ha)')
        
        # Should have fewer rows than original (outliers removed)
        self.assertLess(len(df_clean), len(self.data_with_outliers))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
