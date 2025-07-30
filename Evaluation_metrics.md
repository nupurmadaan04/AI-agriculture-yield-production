# Evaluation Module Documentation

This module provides a standardized, extensible, and modular framework for evaluating regression models using multiple metrics. It is designed to support rapid experimentation, easy metric addition, and robust validation via unit tests.

---

## 📌 Contents

1. [Usage Overview](#usage-overview)
2. [Input/Output Format](#inputoutput-format)
3. [Concrete Metric Classes](#concrete-metric-classes)
4. [Adding a New Metric](#adding-a-new-metric)
5. [Evaluation of Models through Metrics](#evaluation-of-models-through-metrics)
6. [Unit Testing](#unit-testing)
7. [Edge Case Handling](#edge-case-handling)
8. [Code Documentation and Comments](#code-documentation-and-comments)

---

## ✅ Usage Overview

### Step 1: Instantiate Metric and Evaluator Classes

```python
evaluator = Evaluator([R2Metric(), MAEMetric(), RMSEMetric(), MAPEMetric()])
```

### Step 2: Evaluation

Evaluate from csv:
```python
results = evaluator.evaluate_from_csv("path/to/your_file.csv")
```

Evaluate models:
```python
results = evaluator.evaluate_models(models, x_test, y_test, model_names)
```

Save results to csv:
```python
evaluator.save_results_to_csv(results, "path/to/save/results.csv")
```

## 📥 Input/Output Format

### Input CSV format
.csv file containing Actual and Predicted values.
`Actual`: Ground truth values.
`Predicted`: Model output.

### Output JSON Format
```python
{
    "RandomForest": {
        "R2": 0.92,
        "MAE": 120.3,
        "RMSE": 155.6,
        "sMAPE": 4.81
    },
    ...
}
```

## 🔢 Concrete Metric Classes

These are pre-built metrics that inherit from the Metric interface:
- `R2Metric`: Coefficient of Determination.
- `MAEMetric`: Mean Absolute Error.
- `RMSEMetric`: Root Mean Squared Error.
- `MAPEMetric`: Symmetric Mean Absolute Percentage Error.

All metrics implement the following methods:
- `calculate(y_true, y_pred)`
- `name()`
- `description()`

## ➕ Adding a New Metric

### Metric Template

To extend the system with a custom metric, subclass the `Metric` base class:
```python
class NewMetric(Metric):
    def calculate(self, y_true, y_pred):
        # Implement your calculation logic
        return result

    def name(self):
        return "NewMetric"

    def description(self):
        return "Description of what NewMetric measures."
```

### Register the New Metric

Add the new metric to the evaluator:
```python
evaluator = Evaluator([R2Metric(), MAEMetric(), RMSEMetric(), MAPEMetric(), NewMetric()])
```

### Modify Unit Tests

In the `TestMetrics` class, update the `@parameterized.expand` section:
```python
@parameterized.expand([
    ("R2", R2Metric(), r2_score),
    ("MAE", MAEMetric(), mean_absolute_error),
    ("RMSE", RMSEMetric(), lambda y, yp: np.sqrt(mean_squared_error(np.array(y), np.array(yp)))),
    ("sMAPE", MAPEMetric(), lambda y, yp: np.mean(np.abs(np.array(y) - np.array(yp)) / (np.abs(np.array(y)) + np.abs(np.array(yp)) + 1e-10) * 100)),
    ("NewMetric", NewMetric(), your_reference_function)  # <- Add this
])
```

## 📊 Evaluation of Models through Metrics

To evaluate multiple models, add the models' .pkl files to the `models` list and the model names to the `model_names` list.

## 🧪 Unit Testing

Unit tests cover:
- Correct metric computation using reference functions.
- Evaluation pipeline from CSV and model predictions.
- Edge case handling (e.g., empty arrays, NaNs, mismatched lengths).
- CSV result saving.
- Integration with mock models and sample data.

To run the tests in a notebook:
```python
unittest.main(argv=[''], exit=False)
```

## ⚠ Edge Case Handling

Handled in `_validate_inputs()` within metric classes:
- Empty arrays: Raises `ValueError`.
- Mismatched lengths: Raises `ValueError`.
- Non-numeric inputs: Raises `ValueError`.
- NaN or Infinite values: Raises `ValueError`.

In evaluator:
- Missing CSV file: Raises `FileNotFoundError`.
- Failed import: Raises `ImportError`.
- Missing columns: Raises `ValueError`.
- Model prediction failure: Prints a warning without breaking pipeline.

## 🧾 Code Documentation and Comments

- All classes and methods include docstrings explaining:
    - Parameters
    - Return values
    - Purpose
- Key logic is explained via inline comments.
