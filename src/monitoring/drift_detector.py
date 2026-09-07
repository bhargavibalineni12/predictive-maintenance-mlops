import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

def load_prediction_logs(log_file="logs/predictions.log"):
    log_path = Path(log_file)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Prediction log file not found: {log_file}"
        )

    records = []

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Prediction log file is empty.")

    feature_records = [record["input"] for record in records]

    return pd.DataFrame(feature_records)
def load_reference_data(
    data_file="data/raw/engine_data.csv",
):
    data_path = Path(data_file)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Reference data file not found: {data_file}"
        )

    data = pd.read_csv(data_path)

    feature_columns = [
        "Engine rpm",
        "Lub oil pressure",
        "Fuel pressure",
        "Coolant pressure",
        "lub oil temp",
        "Coolant temp",
    ]

    return data[feature_columns]
def validate_feature_columns(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
):
    reference_columns = set(reference_data.columns)
    current_columns = set(current_data.columns)

    if reference_columns != current_columns:
        missing_columns = reference_columns - current_columns
        extra_columns = current_columns - reference_columns

        raise ValueError(
            f"Feature mismatch. "
            f"Missing: {missing_columns}, "
            f"Extra: {extra_columns}"
        )

    return True

def calculate_psi(
    reference_series: pd.Series,
    current_series: pd.Series,
    bins: int = 10,
) -> float:

    # Create bin boundaries using the reference data
    bin_edges = np.quantile(
        reference_series,
        np.linspace(0, 1, bins + 1),
    )

    # Remove duplicate boundaries
    bin_edges = np.unique(bin_edges)

    # Make sure extreme production values are included
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_series,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current_series,
        bins=bin_edges,
    )

    reference_percent = reference_counts / len(reference_series)
    current_percent = current_counts / len(current_series)

    # Avoid log(0) and division by zero
    epsilon = 0.0001

    reference_percent = np.clip(
        reference_percent,
        epsilon,
        None,
    )

    current_percent = np.clip(
        current_percent,
        epsilon,
        None,
    )

    psi_values = (
        current_percent - reference_percent
    ) * np.log(
        current_percent / reference_percent
    )

    return float(np.sum(psi_values))

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.2,
    min_samples: int = 500,
) -> dict:

    if len(current_data) < min_samples:
        raise ValueError(
            f"Not enough production samples for drift detection. "
            f"Required: {min_samples}, "
            f"available: {len(current_data)}"
        )

    validate_feature_columns(
        reference_data,
        current_data,
    )

    drift_results = {}

    for feature in reference_data.columns:

        psi_score = calculate_psi(
            reference_data[feature],
            current_data[feature],
        )

        drift_results[feature] = {
            "psi": round(psi_score, 4),
            "drift_detected": psi_score >= threshold,
        }

    return drift_results

def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.2,
    min_samples: int = 500,
) -> dict:

    drift_results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        threshold=threshold,
        min_samples=min_samples,
    )

    drifted_features = [
        feature
        for feature, result in drift_results.items()
        if result["drift_detected"]
    ]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_status": (
            "DRIFT_DETECTED"
            if drifted_features
            else "NO_DRIFT"
        ),
        "sample_count": len(current_data),
        "psi_threshold": threshold,
        "drifted_features": drifted_features,
        "feature_results": drift_results,
    }

    return report

def save_drift_report(
    report: dict,
    output_dir="artifacts/monitoring",
) -> Path:

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    report_file = output_path / f"drift_report_{timestamp}.json"

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    return report_file