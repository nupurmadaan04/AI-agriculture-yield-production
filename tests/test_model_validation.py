import pytest
import numpy as np
from src.model_validation import model_validation_engine

def test_evaluate_predictions_perfect():
    y_true = np.array([2000.0, 2500.0, 3000.0])
    y_pred = np.array([2000.0, 2500.0, 3000.0])
    res = model_validation_engine.evaluate_predictions(y_true, y_pred)
    assert res['sample_count'] == 3
    assert res['mae'] == 0.0
    assert res['rmse'] == 0.0
    assert res['r2'] == 1.0
    assert res['mape'] == 0.0
    assert res['bias_direction'] == 'BALANCED'

def test_evaluate_predictions_empty():
    res = model_validation_engine.evaluate_predictions(np.array([]), np.array([]))
    assert res['sample_count'] == 0
    assert res['mae'] == 0.0

def test_evaluate_predictions_bias_underprediction():
    # Observed is much higher than predicted -> underpredicting
    y_true = np.array([3000.0, 3500.0, 4000.0])
    y_pred = np.array([2500.0, 3000.0, 3500.0])
    res = model_validation_engine.evaluate_predictions(y_true, y_pred)
    assert res['mean_residual'] == 500.0
    assert res['bias_direction'] == 'UNDERPREDICTING'

def test_out_of_time_evaluation_temporal_split():
    eval_df = model_validation_engine.generate_out_of_time_evaluation(test_year_min=2016)
    assert len(eval_df) > 0
    assert eval_df['Year'].min() >= 2016
    assert 'predicted_yield' in eval_df.columns
    assert 'residual' in eval_df.columns
    assert 'absolute_error' in eval_df.columns
    assert 'prediction_spread' in eval_df.columns
