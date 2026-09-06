from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import (
    load_model,
    load_metadata,
    predict_single,
)


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


model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

print("Model loaded successfully.")
print(f"Model version: {metadata['model_version']}")
print(f"Saved threshold: {metadata['threshold']:.4f}")
print(f"Expected features: {metadata['features']}")


sample_input = {
    "Engine rpm": 700,
    "Lub oil pressure": 2.493592,
    "Fuel pressure": 11.790927,
    "Coolant pressure": 3.178981,
    "lub oil temp": 84.144163,
    "Coolant temp": 81.632187,
}


result = predict_single(
    model,
    metadata,
    sample_input,
)

print("\nPrediction result:")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")