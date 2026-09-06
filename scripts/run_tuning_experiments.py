from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.ingestion import load_data
from src.data.validation import (
    validate_columns,
    validate_missing_values,
    validate_target_values,
    validate_numeric_features,
    validate_duplicates,
)
from src.data.preprocessing import split_data

from src.models.train import (
    tune_logistic_regression,
    tune_random_forest,
)
from src.models.evaluate import (
    evaluate_model_cv,
    find_best_threshold_cv,
)
from src.models.experiment import setup_mlflow, log_model_run


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "engine_data.csv"

df = load_data(DATA_PATH)

validate_columns(df)
validate_missing_values(df)
validate_target_values(df)
validate_numeric_features(df)
validate_duplicates(df)

X_train, X_test, y_train, y_test = split_data(df)

setup_mlflow()

tuned_logistic_model = tune_logistic_regression(
    X_train,
    y_train,
)

tuned_logistic_metrics = evaluate_model_cv(
    tuned_logistic_model,
    X_train,
    y_train,
)

print("\nTuned Logistic Regression - 5-Fold Cross-Validation:")
for metric, value in tuned_logistic_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "Tuned Logistic Regression",
    tuned_logistic_model,
    tuned_logistic_metrics,
)

tuned_random_forest_model = tune_random_forest(
    X_train,
    y_train,
)

tuned_random_forest_metrics = evaluate_model_cv(
    tuned_random_forest_model,
    X_train,
    y_train,
)

print("\nTuned Random Forest - 5-Fold Cross-Validation:")
for metric, value in tuned_random_forest_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "Tuned Random Forest",
    tuned_random_forest_model,
    tuned_random_forest_metrics,
)

best_threshold = find_best_threshold_cv(
    tuned_random_forest_model,
    X_train,
    y_train,
)

print(f"\nSelected classification threshold: {best_threshold:.4f}")