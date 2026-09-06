from pathlib import Path

from src.models.predict import (
    load_model,
    load_metadata,
    predict_single,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def test_model_prediction():
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

    assert result["prediction"] in [0, 1]

    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["threshold"] <= 1.0