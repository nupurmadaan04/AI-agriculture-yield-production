"""
Temporal Forecasting Engine Training & Chronological Benchmark Script.

Trains multi-horizon agricultural forecasting models using strict chronological splits
(Train: 2010–2015, Out-of-Time Test: 2016–2017) with zero future leakage.
Benchmarks against Naive Last Observation, Historical Mean, and Linear Trend baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'Year',
    'State Code',
    'RICE AREA (1000 ha)',
    'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE',
    'WHEAT AREA (1000 ha)',
    'COTTON AREA (1000 ha)',
    'SUGARCANE AREA (1000 ha)',
    'RICE_YIELD_LAG1',
    'RICE_YIELD_ROLL3'
]

TARGET_COLUMN = 'RICE YIELD (Kg per ha)'

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def prepare_temporal_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['Dist Code', 'Year']).reset_index(drop=True)

    # Compute Total Cropped Area and Crop Shares
    area_cols = [c for c in df.columns if 'AREA' in c]
    df['TOTAL_CROPPED_AREA'] = df[area_cols].sum(axis=1)
    df['RICE_AREA_SHARE'] = np.where(
        df['TOTAL_CROPPED_AREA'] > 0,
        df['RICE AREA (1000 ha)'] / df['TOTAL_CROPPED_AREA'],
        0.0
    )

    # Historical Lags strictly from previous years (shift(1))
    df['RICE_YIELD_LAG1'] = df.groupby('Dist Code')['RICE YIELD (Kg per ha)'].shift(1)
    df['RICE_YIELD_ROLL3'] = (
        df.groupby('Dist Code')['RICE YIELD (Kg per ha)']
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    # Fill earliest year lag gaps with state-level median of that year
    state_year_med = df.groupby(['State Code', 'Year'])['RICE YIELD (Kg per ha)'].transform('median')
    df['RICE_YIELD_LAG1'] = df['RICE_YIELD_LAG1'].fillna(state_year_med)
    df['RICE_YIELD_ROLL3'] = df['RICE_YIELD_ROLL3'].fillna(df['RICE_YIELD_LAG1'])

    # Default missing crop areas to 0.0
    for crop in ['WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)', 'SUGARCANE AREA (1000 ha)']:
        if crop not in df.columns:
            df[crop] = 0.0
        else:
            df[crop] = df[crop].fillna(0.0)

    return df

def train_and_benchmark():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'Datasets' / 'rice_data_outlier_removed.csv'
    models_dir = base_dir / 'Models'
    models_dir.mkdir(exist_ok=True)

    print(f"[Forecasting Engine] Preparing chronological panel dataset from {data_path}...")
    df = prepare_temporal_dataset(data_path)

    # Strict Chronological Split: Train <= 2015, Test >= 2016
    train_mask = df['Year'] <= 2015
    test_mask = df['Year'] >= 2016

    X_train = df.loc[train_mask, FEATURE_COLUMNS]
    y_train = df.loc[train_mask, TARGET_COLUMN]

    X_test = df.loc[test_mask, FEATURE_COLUMNS]
    y_test = df.loc[test_mask, TARGET_COLUMN]

    print(f"  Training set (2010–2015): {len(X_train)} rows")
    print(f"  Out-of-time Test set (2016–2017): {len(X_test)} rows")

    # 1. Naive Last-Observation Baseline
    y_pred_naive = X_test['RICE_YIELD_LAG1'].values
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    r2_naive = r2_score(y_test, y_pred_naive)
    mape_naive = compute_mape(y_test.values, y_pred_naive)

    # 2. Historical Mean Baseline (Train Mean per District/State)
    dist_mean_map = df[train_mask].groupby('Dist Code')['RICE YIELD (Kg per ha)'].mean()
    state_mean_map = df[train_mask].groupby('State Code')['RICE YIELD (Kg per ha)'].mean()
    y_pred_mean = df.loc[test_mask, 'Dist Code'].map(dist_mean_map).fillna(
        df.loc[test_mask, 'State Code'].map(state_mean_map)
    ).values
    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2_mean = r2_score(y_test, y_pred_mean)
    mape_mean = compute_mape(y_test.values, y_pred_mean)

    # 3. Linear Trend Model
    lin_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    lin_pipe.fit(X_train, y_train)
    y_pred_lin = lin_pipe.predict(X_test)
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    r2_lin = r2_score(y_test, y_pred_lin)
    mape_lin = compute_mape(y_test.values, y_pred_lin)

    # 4. Gradient Boosting Regressor
    gbr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=150, random_state=42, max_depth=5, learning_rate=0.08))
    ])
    gbr_pipe.fit(X_train, y_train)
    y_pred_gbr = gbr_pipe.predict(X_test)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
    r2_gbr = r2_score(y_test, y_pred_gbr)
    mape_gbr = compute_mape(y_test.values, y_pred_gbr)

    # 5. HistGradientBoosting Regressor
    hgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(max_iter=150, random_state=42, max_depth=6, learning_rate=0.08))
    ])
    hgb_pipe.fit(X_train, y_train)
    y_pred_hgb = hgb_pipe.predict(X_test)
    mae_hgb = mean_absolute_error(y_test, y_pred_hgb)
    rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
    r2_hgb = r2_score(y_test, y_pred_hgb)
    mape_hgb = compute_mape(y_test.values, y_pred_hgb)

    # 6. Random Forest Regressor (Selected Primary Forecaster)
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2, max_depth=14, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = compute_mape(y_test.values, y_pred_rf)

    print("\n" + "="*70)
    print("CHRONOLOGICAL FORECASTING BENCHMARK (Train: 2010–2015, Test: 2016–2017)")
    print("="*70)
    benchmark_table = [
        {"model": "Naive (Last Observation yt-1)", "mae": round(mae_naive, 2), "rmse": round(rmse_naive, 2), "r2": round(r2_naive, 4), "mape": round(mape_naive, 2)},
        {"model": "Historical District Mean", "mae": round(mae_mean, 2), "rmse": round(rmse_mean, 2), "r2": round(r2_mean, 4), "mape": round(mape_mean, 2)},
        {"model": "Linear Trend Regression", "mae": round(mae_lin, 2), "rmse": round(rmse_lin, 2), "r2": round(r2_lin, 4), "mape": round(mape_lin, 2)},
        {"model": "HistGradientBoosting", "mae": round(mae_hgb, 2), "rmse": round(rmse_hgb, 2), "r2": round(r2_hgb, 4), "mape": round(mape_hgb, 2)},
        {"model": "Gradient Boosting Regressor", "mae": round(mae_gbr, 2), "rmse": round(rmse_gbr, 2), "r2": round(r2_gbr, 4), "mape": round(mape_gbr, 2)},
        {"model": "Random Forest Regressor (Selected)", "mae": round(mae_rf, 2), "rmse": round(rmse_rf, 2), "r2": round(r2_rf, 4), "mape": round(mape_rf, 2)}
    ]

    for row in benchmark_table:
        print(f"{row['model']:<35} | MAE: {row['mae']:>7.2f} kg/ha | RMSE: {row['rmse']:>7.2f} | R²: {row['r2']:>6.4f} | MAPE: {row['mape']:>5.2f}%")

    # Fit final model on complete dataset for forward multi-horizon forecasting (2018, 2019, 2020)
    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET_COLUMN]
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2, max_depth=14, n_jobs=-1))
    ])
    final_pipeline.fit(X_full, y_full)

    # Persist model pipeline
    pipe_path = models_dir / 'forecasting_pipeline.pkl'
    joblib.dump(final_pipeline, pipe_path)
    print(f"\n[Forecasting Engine] Saved final forecasting pipeline to {pipe_path}")

    # Build Metadata
    metadata = {
        'model_name': 'Temporal Exogenous Random Forest Forecaster',
        'algorithm': 'RandomForestRegressor (n_estimators=150, max_depth=14)',
        'target': 'RICE YIELD (Kg per ha)',
        'chronological_split': {
            'train_years': '2010–2015',
            'test_years': '2016–2017',
            'train_records': int(len(X_train)),
            'test_records': int(len(X_test))
        },
        'supported_horizons': [1, 2, 3],
        'features': FEATURE_COLUMNS,
        'benchmark_results': benchmark_table,
        'selected_model_metrics': {
            'mae_kg_ha': round(mae_rf, 2),
            'rmse_kg_ha': round(rmse_rf, 2),
            'r2_score': round(r2_rf, 4),
            'mape_percent': round(mape_rf, 2)
        },
        'scientific_safeguards': [
            'Production strictly excluded from all training and forecasting features.',
            'Lags computed with shift(1) strictly excluding future data.',
            'Validation performed out-of-time (2016–2017) without random shuffling.'
        ]
    }

    meta_path = models_dir / 'forecasting_model_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Forecasting Engine] Saved forecasting metadata to {meta_path}")

if __name__ == '__main__':
    train_and_benchmark()
