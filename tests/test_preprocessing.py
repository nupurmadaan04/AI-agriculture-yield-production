"""
Unit tests for preprocessing functions.

This module tests all the preprocessing functions for the agriculture yield prediction project.
Each test covers different scenarios like missing data, outliers, scaling, and data validation.
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
import preprocessing_functions as pf


class TestPreprocessingFunctions(unittest.TestCase):
    """Test cases for all preprocessing functions."""
    
    def setUp(self):
        """Set up test data before each test."""
        # Create sample crop data
        self.sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024],
            'State Name': ['Punjab', 'Haryana', 'Punjab', 'West Bengal', 'Haryana'],
            'Dist Name': ['District A', 'District B', 'District C', 'District D', 'District E'],
            'RICE AREA (1000 ha)': [100.5, 150.2, 120.8, 80.3, 200.1],
            'RICE PRODUCTION (1000 tons)': [300.0, 450.5, 360.2, 240.8, 600.3],
            'RICE YIELD (Kg per ha)': [2980, 3000, 2990, 3005, 3002],
            'OTHER_CROP': [50, 60, 55, 45, 70]  # Extra column
        })
        
        # Create data with missing values
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[0, 'RICE AREA (1000 ha)'] = np.nan
        self.data_with_missing.loc[1, 'RICE YIELD (Kg per ha)'] = np.nan
        
        # Create data with outliers
        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[0, 'RICE YIELD (Kg per ha)'] = 10000  # Very high outlier
        self.data_with_outliers.loc[1, 'RICE YIELD (Kg per ha)'] = 500    # Very low outlier
        
        # Create temporary directory for file tests
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_crops.csv')
        self.sample_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    # ==================== Data Loading Tests ====================
    
    def test_load_crop_data_success(self):
        """Test successful loading of crop data."""
        result = pf.load_crop_data(self.test_file)
        
        # Check if data was loaded correctly
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result.columns), list(self.sample_data.columns))
    
    def test_load_crop_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            pf.load_crop_data('/nonexistent/path/file.csv')
    
    def test_load_crop_data_empty_file(self):
        """Test loading empty CSV file."""
        empty_file = os.path.join(self.temp_dir, 'empty.csv')
        # Create a truly empty file
        with open(empty_file, 'w') as f:
            f.write('')  # Empty file
        
        with self.assertRaises(ValueError):
            pf.load_crop_data(empty_file)
    
    # ==================== Data Filtering Tests ====================
    
    def test_filter_rice_data_success(self):
        """Test successful rice data filtering."""
        result = pf.filter_rice_data(self.sample_data)
        
        expected_columns = [
            'Year', 'State Name', 'Dist Name',
            'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 
            'RICE YIELD (Kg per ha)'
        ]
        
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 5)
        # Should not include 'OTHER_CROP' column
        self.assertNotIn('OTHER_CROP', result.columns)
    
    def test_filter_rice_data_missing_columns(self):
        """Test filtering when required columns are missing."""
        incomplete_data = pd.DataFrame({
            'Year': [2020, 2021],
            'State Name': ['Punjab', 'Haryana']
            # Missing other required columns
        })
        
        with self.assertRaises(KeyError) as context:
            pf.filter_rice_data(incomplete_data)
        
        self.assertIn('Missing required columns', str(context.exception))
    
    # ==================== Missing Values Tests ====================
    
    def test_check_missing_values(self):
        """Test checking for missing values."""
        missing_count = pf.check_missing_values(self.data_with_missing)
        
        # Should detect missing values
        self.assertEqual(missing_count['RICE AREA (1000 ha)'], 1)
        self.assertEqual(missing_count['RICE YIELD (Kg per ha)'], 1)
        self.assertEqual(missing_count['Year'], 0)  # No missing values in Year
    
    def test_handle_missing_values_drop(self):
        """Test handling missing values by dropping rows."""
        result = pf.handle_missing_values(self.data_with_missing, strategy='drop')
        
        # Should have fewer rows after dropping
        self.assertLess(len(result), len(self.data_with_missing))
        # Should have no missing values
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_handle_missing_values_fill_mean(self):
        """Test handling missing values by filling with mean."""
        result = pf.handle_missing_values(self.data_with_missing, strategy='fill_mean')
        
        # Should have same number of rows
        self.assertEqual(len(result), len(self.data_with_missing))
        # Numeric columns should have no missing values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertEqual(result[numeric_cols].isnull().sum().sum(), 0)
    
    def test_handle_missing_values_fill_median(self):
        """Test handling missing values by filling with median."""
        result = pf.handle_missing_values(self.data_with_missing, strategy='fill_median')
        
        # Should have same number of rows
        self.assertEqual(len(result), len(self.data_with_missing))
        # Numeric columns should have no missing values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertEqual(result[numeric_cols].isnull().sum().sum(), 0)
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test handling missing values with invalid strategy."""
        with self.assertRaises(ValueError):
            pf.handle_missing_values(self.sample_data, strategy='invalid_method')
    
    # ==================== Categorical Encoding Tests ====================
    
    def test_encode_categorical_column_success(self):
        """Test successful categorical encoding."""
        result = pf.encode_categorical_column(self.sample_data, 'State Name')
        
        # Should have new encoded column
        self.assertIn('state_name_encoded', result.columns)
        # Original column should still exist
        self.assertIn('State Name', result.columns)
        # Encoded column should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result['state_name_encoded']))
        # Should have same number of rows
        self.assertEqual(len(result), len(self.sample_data))
    
    def test_encode_categorical_column_missing_column(self):
        """Test encoding non-existent column."""
        with self.assertRaises(KeyError):
            pf.encode_categorical_column(self.sample_data, 'NonExistent Column')
    
    def test_encode_categorical_column_with_nan(self):
        """Test encoding column with NaN values."""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, 'State Name'] = np.nan
        
        result = pf.encode_categorical_column(data_with_nan, 'State Name')
        
        # Should handle NaN by replacing with 'Unknown'
        self.assertIn('state_name_encoded', result.columns)
        self.assertFalse(result['state_name_encoded'].isnull().any())
    
    # ==================== Outlier Detection Tests ====================
    
    def test_detect_outliers_iqr_success(self):
        """Test successful outlier detection."""
        outlier_info = pf.detect_outliers_iqr(self.data_with_outliers, 'RICE YIELD (Kg per ha)')
        
        # Should return dictionary with required keys
        required_keys = ['lower_bound', 'upper_bound', 'Q1', 'Q3', 'IQR', 'outlier_count', 'outlier_indices']
        for key in required_keys:
            self.assertIn(key, outlier_info)
        
        # Should detect outliers
        self.assertGreater(outlier_info['outlier_count'], 0)
        # IQR should be reasonable
        self.assertGreaterEqual(outlier_info['IQR'], 0)
    
    def test_detect_outliers_iqr_missing_column(self):
        """Test outlier detection with non-existent column."""
        with self.assertRaises(KeyError):
            pf.detect_outliers_iqr(self.sample_data, 'NonExistent Column')
    
    def test_detect_outliers_iqr_non_numeric_column(self):
        """Test outlier detection with non-numeric column."""
        with self.assertRaises(ValueError):
            pf.detect_outliers_iqr(self.sample_data, 'State Name')
    
    def test_remove_outliers_iqr_success(self):
        """Test successful outlier removal."""
        result = pf.remove_outliers_iqr(self.data_with_outliers, 'RICE YIELD (Kg per ha)')
        
        # Should have fewer rows after removing outliers
        self.assertLess(len(result), len(self.data_with_outliers))
        # Extreme values should be removed
        self.assertNotIn(10000, result['RICE YIELD (Kg per ha)'].values)
        self.assertNotIn(500, result['RICE YIELD (Kg per ha)'].values)
    
    # ==================== Feature Scaling Tests ====================
    
    def test_scale_features_success(self):
        """Test successful feature scaling."""
        # Prepare training and test data
        X_train = self.sample_data[['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']]
        X_test = X_train.iloc[:2].copy()  # First 2 rows as test
        
        X_train_scaled, X_test_scaled, scaler = pf.scale_features(X_train, X_test)
        
        # Should return numpy arrays
        self.assertIsInstance(X_train_scaled, np.ndarray)
        self.assertIsInstance(X_test_scaled, np.ndarray)
        # Should have same shapes
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        # Training data should have approximately zero mean and unit variance
        np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_scale_features_train_only(self):
        """Test feature scaling with only training data."""
        X_train = self.sample_data[['Year', 'RICE AREA (1000 ha)']]
        
        X_train_scaled, X_test_scaled, scaler = pf.scale_features(X_train)
        
        self.assertIsInstance(X_train_scaled, np.ndarray)
        self.assertIsNone(X_test_scaled)
        self.assertIsNotNone(scaler)
    
    def test_scale_features_empty_data(self):
        """Test feature scaling with empty data."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            pf.scale_features(empty_df)
    
    # ==================== Data Splitting Tests ====================
    
    def test_split_data_success(self):
        """Test successful data splitting."""
        X = self.sample_data[['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']]
        y = self.sample_data['RICE YIELD (Kg per ha)']
        
        X_train, X_test, y_train, y_test = pf.split_data(X, y, test_size=0.4, random_state=42)
        
        # Check shapes (40% test = 2 rows, 60% train = 3 rows)
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(y_test), 2)
    
    def test_split_data_invalid_test_size(self):
        """Test data splitting with invalid test size."""
        X = self.sample_data[['Year']]
        y = self.sample_data['RICE YIELD (Kg per ha)']
        
        # Test size > 1
        with self.assertRaises(ValueError):
            pf.split_data(X, y, test_size=1.5)
        
        # Test size = 0
        with self.assertRaises(ValueError):
            pf.split_data(X, y, test_size=0)
    
    def test_split_data_mismatched_lengths(self):
        """Test data splitting with mismatched X and y lengths."""
        X = self.sample_data[['Year']]
        y = self.sample_data['RICE YIELD (Kg per ha)'].iloc[:3]  # Different length
        
        with self.assertRaises(ValueError):
            pf.split_data(X, y)
    
    def test_split_data_empty_data(self):
        """Test data splitting with empty data."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        
        with self.assertRaises(ValueError):
            pf.split_data(X, y)
    
    # ==================== Data Validation Tests ====================
    
    def test_validate_data_format_success(self):
        """Test successful data format validation."""
        validation = pf.validate_data_format(self.sample_data)
        
        # Check required keys
        required_keys = ['shape', 'columns', 'dtypes', 'missing_values', 'duplicate_rows', 'memory_usage_mb']
        for key in required_keys:
            self.assertIn(key, validation)
        
        # Check specific values
        self.assertEqual(validation['shape'], (5, 7))  # 5 rows, 7 columns
        self.assertEqual(len(validation['columns']), 7)
        self.assertIsInstance(validation['memory_usage_mb'], float)
    
    def test_validate_data_format_with_problems(self):
        """Test data validation with problematic data."""
        # Create problematic data
        bad_data = self.sample_data.copy()
        bad_data.loc[0, 'RICE YIELD (Kg per ha)'] = -100  # Negative yield
        bad_data.loc[1, 'RICE AREA (1000 ha)'] = 0        # Zero area
        bad_data.loc[2, 'RICE PRODUCTION (1000 tons)'] = -50  # Negative production
        
        validation = pf.validate_data_format(bad_data)
        
        # Should detect data quality issues
        self.assertEqual(validation['negative_yield_count'], 1)
        self.assertEqual(validation['zero_area_count'], 1)
        self.assertEqual(validation['negative_production_count'], 1)
    
    # ==================== File Operations Tests ====================
    
    def test_save_processed_data_success(self):
        """Test successful data saving."""
        output_file = os.path.join(self.temp_dir, 'processed_data.csv')
        
        # Should save without error
        pf.save_processed_data(self.sample_data, output_file)
        
        # File should exist
        self.assertTrue(os.path.exists(output_file))
        
        # Should be able to load the saved data
        loaded_data = pd.read_csv(output_file)
        self.assertEqual(len(loaded_data), len(self.sample_data))
    
    def test_save_processed_data_empty_dataframe(self):
        """Test saving empty dataframe."""
        empty_df = pd.DataFrame()
        output_file = os.path.join(self.temp_dir, 'empty.csv')
        
        with self.assertRaises(ValueError):
            pf.save_processed_data(empty_df, output_file)
    
    # ==================== Complete Pipeline Test ====================
    
    def test_complete_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        input_file = self.test_file
        output_file = os.path.join(self.temp_dir, 'processed_output.csv')
        
        # Run complete pipeline
        summary = pf.complete_preprocessing_pipeline(input_file, output_file)
        
        # Check summary contains expected keys
        expected_keys = ['original_shape', 'final_shape', 'rows_removed', 'columns_added', 'validation_summary', 'output_file']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Load and check processed data
        processed_data = pd.read_csv(output_file)
        self.assertGreater(len(processed_data), 0)
        self.assertIn('state_name_encoded', processed_data.columns)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
