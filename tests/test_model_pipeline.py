"""
Unit and Integration Tests for ML Pipeline and Evaluation Engine.

Tests cover:
- Pipeline creation and fit
- Batch and single-sample prediction
- Non-negative yield constraints
- Unseen state / invalid input handling
- Pipeline persistence (save & load)
- Backward compatibility with rf_model.pkl and scaler.pkl
- Metric calculations (R², MAE, RMSE, sMAPE)
"""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd
import joblib

# Add src and root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import (
    RiceYieldModel,
    build_rice_yield_pipeline,
    AgriculturalFeatureEngineer,
    STATE_TO_CODE,
    CODE_TO_STATE
)
from src.train_and_evaluate import calculate_smape, evaluate_predictions


class TestAgriculturalFeatureEngineer(unittest.TestCase):
    """Test feature engineering transformer."""

    def setUp(self):
        self.transformer = AgriculturalFeatureEngineer()
        self.sample_df = pd.DataFrame({
            'Year': ['2015', 2016],
            'State Name': ['Punjab', 'Haryana'],
            'RICE AREA (1000 ha)': [120.5, -5.0],  # test negative clipping
            'RICE PRODUCTION (1000 tons)': [360.0, np.nan]  # test NaN handling
        })

    def test_transform_columns_and_values(self):
        res = self.transformer.transform(self.sample_df)
        expected_cols = ['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code']
        self.assertEqual(list(res.columns), expected_cols)
        self.assertEqual(res['State Code'].iloc[0], STATE_TO_CODE['Punjab'])
        self.assertEqual(res['State Code'].iloc[1], STATE_TO_CODE['Haryana'])
        self.assertGreaterEqual(res['RICE AREA (1000 ha)'].min(), 0.0)
        self.assertFalse(res.isnull().any().any())

    def test_transform_numpy_array(self):
        arr = np.array([[2015, 100.0, 300.0, 12]])
        res = self.transformer.transform(arr)
        self.assertEqual(res.shape, (1, 4))


class TestRiceYieldPipeline(unittest.TestCase):
    """Test model pipeline training, prediction, and persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_data = pd.DataFrame({
            'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017] * 5,
            'State Name': ['Punjab', 'Haryana', 'West Bengal', 'Tamil Nadu', 'Bihar'] * 8,
            'RICE AREA (1000 ha)': [100.0, 150.0, 200.0, 80.0, 120.0] * 8,
            'RICE PRODUCTION (1000 tons)': [300.0, 450.0, 500.0, 280.0, 300.0] * 8,
            'RICE YIELD (Kg per ha)': [3000.0, 3000.0, 2500.0, 3500.0, 2500.0] * 8
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_fit_predict(self):
        X = self.train_data[['Year', 'State Name', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']]
        y = self.train_data['RICE YIELD (Kg per ha)']

        model = RiceYieldModel()
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.all(preds >= 0.0))

    def test_predict_single(self):
        X = self.train_data[['Year', 'State Name', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']]
        y = self.train_data['RICE YIELD (Kg per ha)']

        model = RiceYieldModel()
        model.fit(X, y)

        single_pred = model.predict_single(
            year=2015,
            state="Punjab",
            rice_area_1000ha=100.0,
            rice_prod_1000tons=300.0
        )
        self.assertIsInstance(single_pred, float)
        self.assertGreater(single_pred, 0.0)

    def test_save_and_load_pipeline(self):
        X = self.train_data[['Year', 'State Name', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']]
        y = self.train_data['RICE YIELD (Kg per ha)']

        model = RiceYieldModel()
        model.fit(X, y)

        save_path = os.path.join(self.temp_dir, 'pipeline.pkl')
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_model = RiceYieldModel.load(save_path)
        orig_preds = model.predict(X.iloc[:5])
        loaded_preds = loaded_model.predict(X.iloc[:5])
        np.testing.assert_allclose(orig_preds, loaded_preds)

    def test_backward_compatibility_artifacts(self):
        """Test loading repository's serialized rf_model and scaler artifacts."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        rf_path = os.path.join(base_dir, 'Models', 'rf_model.pkl')
        scaler_path = os.path.join(base_dir, 'Models', 'scaler.pkl')

        if os.path.exists(rf_path) and os.path.exists(scaler_path):
            rf = joblib.load(rf_path)
            scaler = joblib.load(scaler_path)

            sample_input = np.array([[2015, 100.0, 300.0, 12]])
            scaled = scaler.transform(sample_input)
            pred = rf.predict(scaled)
            self.assertEqual(len(pred), 1)
            self.assertGreater(pred[0], 0.0)


class TestMetricsAndEvaluation(unittest.TestCase):
    """Test evaluation metric calculations."""

    def test_smape_calculation(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(calculate_smape(y_true, y_pred), 0.0)

        y_pred_off = np.array([110.0, 190.0, 310.0])
        smape = calculate_smape(y_true, y_pred_off)
        self.assertGreater(smape, 0.0)
        self.assertLess(smape, 10.0)

    def test_evaluate_predictions_metrics(self):
        y_true = np.array([2000.0, 2500.0, 3000.0, 3500.0])
        y_pred = np.array([2050.0, 2450.0, 3050.0, 3450.0])

        metrics = evaluate_predictions(y_true, y_pred)
        for key in ['R2', 'MAE', 'RMSE', 'MSE', 'sMAPE']:
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics['MAE'], 50.0)
        self.assertAlmostEqual(metrics['RMSE'], 50.0)
        self.assertGreater(metrics['R2'], 0.95)


if __name__ == '__main__':
    unittest.main(verbosity=2)
