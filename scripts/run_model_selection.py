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
    train_selected_random_forest,
    save_model,
    save_model_metadata,
)
from src.models.evaluate import (
    find_best_threshold_cv,
    evaluate_final_test,
)


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "engine_data.csv"


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

df = load_data(DATA_PATH)

validate_columns(df)
validate_missing_values(df)
validate_target_values(df)
validate_numeric_features(df)
validate_duplicates(df)


# ---------------------------------------------------------
# Train / test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = split_data(df)


# ---------------------------------------------------------
# Train selected model
# ---------------------------------------------------------

selected_model = train_selected_random_forest(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Select threshold using training data only
# ---------------------------------------------------------

best_threshold = find_best_threshold_cv(
    selected_model,
    X_train,
    y_train,
)

print(f"\nLocked classification threshold: {best_threshold:.4f}")

# ---------------------------------------------------------
# Final evaluation on untouched test set
# ---------------------------------------------------------

final_metrics, confusion_matrix_result = evaluate_final_test(
    selected_model,
    X_test,
    y_test,
    best_threshold,
)

# ---------------------------------------------------------
# Save final model artifact
# ---------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "model"
    / "random_forest_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
    / "model_metadata.json"
)

save_model(
    selected_model,
    MODEL_PATH,
)


# ---------------------------------------------------------
# Save model metadata
# ---------------------------------------------------------

metadata = {
    "model_name": "RandomForestClassifier",
    "model_version": "1.0",
    "threshold": float(best_threshold),
    "features": list(X_train.columns),
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "random_state": 42,
    },
    "final_test_metrics": {
        metric: float(value)
        for metric, value in final_metrics.items()
    },
}

save_model_metadata(
    metadata,
    METADATA_PATH,
)