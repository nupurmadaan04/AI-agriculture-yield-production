"""
Model Experimentation, Benchmarking, Error Analysis, and Explainability Engine.

This script executes a rigorous machine learning pipeline comparing multiple models,
evaluating them under both Random Split and Temporal Split, generating detailed error analysis,
computing feature importance, and serializing the best model pipeline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

# Import custom pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import RiceYieldModel, build_rice_yield_pipeline, STATE_TO_CODE


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # Handle zeros gracefully
    valid = denominator > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(diff[valid] / denominator[valid]) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression evaluation metrics."""
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'sMAPE': calculate_smape(y_true, y_pred)
    }


def load_dataset(dataset_path: str = "Datasets/rice_data_outlier_removed.csv") -> pd.DataFrame:
    """Load cleaned rice dataset with validation."""
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / dataset_path
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {full_path}")
    df = pd.read_csv(full_path)
    return df


def run_model_benchmarks(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], RiceYieldModel]:
    """
    Run systematic benchmark across regression algorithms under both Random and Temporal splits.
    """
    features = ['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code']
    target = 'RICE YIELD (Kg per ha)'

    X = df[features].copy()
    y = df[target].copy()

    # 1. Random 80/20 Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Temporal Holdout Split (Train <= 2015, Test > 2015)
    train_mask = df['Year'] <= 2015
    test_mask = df['Year'] > 2015
    X_train_t = X[train_mask]
    y_train_t = y[train_mask]
    X_test_t = X[test_mask]
    y_test_t = y[test_mask]

    models_to_test = {
        'RandomForest (Baseline)': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'RandomForest (Tuned)': RandomForestRegressor(
            n_estimators=150, max_depth=30, min_samples_split=2, random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=150, max_depth=30, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=150, max_depth=10, random_state=42
        ),
        'SVR (RBF)': SVR(
            kernel='rbf', C=1000.0, epsilon=0.1
        )
    }

    results = []
    trained_pipelines: Dict[str, Pipeline] = {}

    print(f"{'='*80}\nBENCHMARKING REGRESSION MODELS\n{'='*80}")

    for name, regressor in models_to_test.items():
        # Test on Random Split
        pipe = build_rice_yield_pipeline(regressor=regressor, random_state=42)
        
        t0 = time.perf_counter()
        pipe.fit(X_train_r, y_train_r)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        train_preds_r = np.clip(pipe.predict(X_train_r), 0, None)
        test_preds_r = np.clip(pipe.predict(X_test_r), 0, None)
        infer_time = time.perf_counter() - t0

        train_metrics_r = evaluate_predictions(y_train_r.values, train_preds_r)
        test_metrics_r = evaluate_predictions(y_test_r.values, test_preds_r)

        # Cross Validation on Random Train split (5 folds)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            pipe, X_train_r, y_train_r, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            n_jobs=-1
        )
        cv_r2 = float(np.mean(cv_scores['test_r2']))
        cv_mae = float(-np.mean(cv_scores['test_neg_mean_absolute_error']))
        cv_rmse = float(-np.mean(cv_scores['test_neg_root_mean_squared_error']))

        # Test on Temporal Split
        pipe_t = build_rice_yield_pipeline(regressor=regressor, random_state=42)
        pipe_t.fit(X_train_t, y_train_t)
        test_preds_t = np.clip(pipe_t.predict(X_test_t), 0, None)
        test_metrics_t = evaluate_predictions(y_test_t.values, test_preds_t)

        results.append({
            'Model': name,
            'Train R2': round(train_metrics_r['R2'], 4),
            'Test R2 (Random)': round(test_metrics_r['R2'], 4),
            'Test MAE (Random)': round(test_metrics_r['MAE'], 2),
            'Test RMSE (Random)': round(test_metrics_r['RMSE'], 2),
            'Test sMAPE (Random) %': round(test_metrics_r['sMAPE'], 2),
            '5-Fold CV R2': round(cv_r2, 4),
            '5-Fold CV MAE': round(cv_mae, 2),
            '5-Fold CV RMSE': round(cv_rmse, 2),
            'Test R2 (Temporal)': round(test_metrics_t['R2'], 4),
            'Test MAE (Temporal)': round(test_metrics_t['MAE'], 2),
            'Test RMSE (Temporal)': round(test_metrics_t['RMSE'], 2),
            'Test sMAPE (Temporal) %': round(test_metrics_t['sMAPE'], 2),
            'Training Time (s)': round(train_time, 3),
            'Inference Latency (s)': round(infer_time, 4)
        })
        trained_pipelines[name] = pipe

        print(f"[{name}] Test R² (Random): {test_metrics_r['R2']:.4f} | MAE: {test_metrics_r['MAE']:.2f} | RMSE: {test_metrics_r['RMSE']:.2f} | Temporal R²: {test_metrics_t['R2']:.4f}")

    results_df = pd.DataFrame(results)

    # Save best model pipeline
    best_pipe = trained_pipelines['RandomForest (Tuned)']
    best_model = RiceYieldModel(pipeline=best_pipe)
    best_model.is_fitted = True

    return results_df, {
        'X_test': X_test_r,
        'y_test': y_test_r,
        'X_train': X_train_r,
        'y_train': y_train_r,
        'df_full': df
    }, best_model


def perform_error_analysis(
    model: RiceYieldModel,
    test_data: Dict[str, Any],
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate comprehensive error analysis dataset and error summary breakdowns.
    """
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    df_full = test_data['df_full'].loc[y_test.index].copy()

    preds = model.predict(X_test)
    y_actual = y_test.values

    residuals = y_actual - preds
    abs_errors = np.abs(residuals)
    pct_errors = np.where(y_actual > 0, (abs_errors / y_actual) * 100.0, 0.0)

    error_df = pd.DataFrame({
        'State Name': df_full['State Name'].values,
        'State Code': df_full['State Code'].values,
        'Dist Name': df_full['Dist Name'].values,
        'Year': df_full['Year'].values,
        'Rice Area (1000 ha)': df_full['RICE AREA (1000 ha)'].values,
        'Rice Production (1000 tons)': df_full['RICE PRODUCTION (1000 tons)'].values,
        'Actual Yield (Kg/ha)': y_actual,
        'Predicted Yield (Kg/ha)': np.round(preds, 2),
        'Residual (Kg/ha)': np.round(residuals, 2),
        'Absolute Error (Kg/ha)': np.round(abs_errors, 2),
        'Percentage Error (%)': np.round(pct_errors, 2)
    })

    # Save to CSV
    error_csv_path = output_dir / 'error_analysis.csv'
    error_df.to_csv(error_csv_path, index=False)
    print(f"Error analysis saved to {error_csv_path}")

    # Generate error breakdown by State
    state_errors = error_df.groupby('State Name').agg(
        Count=('Actual Yield (Kg/ha)', 'count'),
        MAE=('Absolute Error (Kg/ha)', 'mean'),
        RMSE=('Residual (Kg/ha)', lambda x: np.sqrt(np.mean(x**2))),
        Mean_Percentage_Error=('Percentage Error (%)', 'mean')
    ).reset_index().sort_values(by='MAE', ascending=False)

    state_errors_path = output_dir / 'error_by_state.csv'
    state_errors.to_csv(state_errors_path, index=False)

    # Generate diagnostic plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, preds, alpha=0.6, color='#2b5c8f', edgecolors='k', s=35)
    max_val = max(y_actual.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')
    plt.xlabel('Actual Rice Yield (Kg/ha)', fontsize=12)
    plt.ylabel('Predicted Rice Yield (Kg/ha)', fontsize=12)
    plt.title('Actual vs. Predicted Rice Yield', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / 'actual_vs_predicted.png', dpi=200)
    plt.close()

    # 2. Residual Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#2a9d8f', bins=30)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Residual Error (Actual - Predicted) [Kg/ha]', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / 'residual_distribution.png', dpi=200)
    plt.close()

    # 3. Residual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.6, color='#e76f51', edgecolors='k', s=35)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.xlabel('Predicted Rice Yield (Kg/ha)', fontsize=12)
    plt.ylabel('Residual (Kg/ha)', fontsize=12)
    plt.title('Residuals vs. Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / 'residuals_vs_fitted.png', dpi=200)
    plt.close()

    return error_df


def perform_explainability_analysis(
    model: RiceYieldModel,
    test_data: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Extract native tree feature importance and permutation feature importance.
    """
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    pipeline = model.pipeline

    regressor = pipeline.named_steps['regressor']
    feature_names = ['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code']

    # Native importance (MDI)
    native_importances = {}
    if hasattr(regressor, 'feature_importances_'):
        for name, imp in zip(feature_names, regressor.feature_importances_):
            native_importances[name] = float(imp)

    # Permutation importance on Test set
    perm = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring='r2'
    )
    perm_importances = {}
    for name, mean_imp, std_imp in zip(feature_names, perm.importances_mean, perm.importances_std):
        perm_importances[name] = {
            'mean_importance': float(mean_imp),
            'std_importance': float(std_imp)
        }

    explainability_data = {
        'model_type': type(regressor).__name__,
        'native_feature_importance': native_importances,
        'permutation_feature_importance': perm_importances,
        'explanation_notes': [
            "Feature importance indicates how heavily the model relies on specific input features to make predictions.",
            "Higher importance reflects significant contribution to reducing variance in yield predictions.",
            "RICE PRODUCTION and RICE AREA are the primary drivers as they approximate the biological production/area relationship.",
            "State Code captures regional agro-climatic baseline variations."
        ]
    }

    json_path = output_dir / 'feature_importance.json'
    with open(json_path, 'w') as f:
        json.dump(explainability_data, f, indent=4)
    print(f"Explainability metrics saved to {json_path}")

    # Plot feature importance
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(8, 5))
    sorted_features = sorted(native_importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    sns.barplot(x=values, y=names, palette='viridis')
    plt.xlabel('Relative Feature Importance (MDI)', fontsize=12)
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=200)
    plt.close()

    return explainability_data


def main():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / 'Models'
    models_dir.mkdir(exist_ok=True, parents=True)

    df = load_dataset("Datasets/rice_data_outlier_removed.csv")
    print(f"Loaded dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Benchmark models
    comparison_df, test_data, best_model = run_model_benchmarks(df)

    # Save model comparison table
    comparison_csv_path = models_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison table saved to {comparison_csv_path}")
    print("\n" + comparison_df.to_string(index=False))

    # 2. Save best pipeline & backward compatible artifacts
    pipeline_path = models_dir / 'rf_pipeline.pkl'
    best_model.save(pipeline_path)
    print(f"\nBest model pipeline saved to {pipeline_path}")

    # Backward compatibility artifacts
    raw_rf = best_model.pipeline.named_steps['regressor']
    raw_scaler = best_model.pipeline.named_steps['scaler']
    joblib.dump(raw_rf, models_dir / 'rf_model.pkl')
    joblib.dump(raw_scaler, models_dir / 'scaler.pkl')

    # Save test predictions CSV
    preds = best_model.predict(test_data['X_test'])
    test_pred_df = pd.DataFrame({
        'Actual': test_data['y_test'].values,
        'Predicted': preds
    })
    test_pred_df.to_csv(models_dir / 'test_predictions.csv', index=False)

    # 3. Error Analysis
    perform_error_analysis(best_model, test_data, models_dir)

    # 4. Explainability Analysis
    perform_explainability_analysis(best_model, test_data, models_dir)

    print("\nAll model training, benchmarking, error analysis, and explainability workflows completed successfully!")


if __name__ == '__main__':
    main()
