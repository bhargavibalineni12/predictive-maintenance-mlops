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
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
)

from src.models.evaluate import evaluate_model_cv
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

logistic_model = train_logistic_regression(X_train, y_train)

logistic_metrics = evaluate_model_cv(
    logistic_model,
    X_train,
    y_train,
)

print("\nLogistic Regression - 5-Fold Cross-Validation:")
for metric, value in logistic_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "Logistic Regression",
    logistic_model,
    logistic_metrics,
)

# =========================================================
# DECISION TREE
# =========================================================

decision_tree_model = train_decision_tree(X_train, y_train)

decision_tree_metrics = evaluate_model_cv(
    decision_tree_model,
    X_train,
    y_train,
)

print("\nDecision Tree - 5-Fold Cross-Validation:")
for metric, value in decision_tree_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "Decision Tree",
    decision_tree_model,
    decision_tree_metrics,
)


# =========================================================
# RANDOM FOREST
# =========================================================

random_forest_model = train_random_forest(X_train, y_train)

random_forest_metrics = evaluate_model_cv(
    random_forest_model,
    X_train,
    y_train,
)

print("\nRandom Forest - 5-Fold Cross-Validation:")
for metric, value in random_forest_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "Random Forest",
    random_forest_model,
    random_forest_metrics,
)


# =========================================================
# XGBOOST
# =========================================================

xgboost_model = train_xgboost(X_train, y_train)

xgboost_metrics = evaluate_model_cv(
    xgboost_model,
    X_train,
    y_train,
)

print("\nXGBoost - 5-Fold Cross-Validation:")
for metric, value in xgboost_metrics.items():
    print(f"{metric}: {value:.4f}")

log_model_run(
    "XGBoost",
    xgboost_model,
    xgboost_metrics,
)