from pathlib import Path
import json
import joblib
import pandas as pd


def load_model(model_path):
    """
    Load a trained model artifact.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model


def load_metadata(metadata_path):
    """
    Load model metadata.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata artifact not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return metadata


def predict_single(
    model,
    metadata,
    input_data,
):
    """
    Generate a prediction for one observation using
    the saved model and locked classification threshold.
    """

    feature_names = metadata["features"]
    threshold = metadata["threshold"]

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names,
    )

    probability = model.predict_proba(input_df)[0, 1]

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold": float(threshold),
    }