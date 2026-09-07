import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import (
    detect_drift,
    generate_drift_report,
    save_drift_report,
)

def test_detect_drift_with_insufficient_samples():
    np.random.seed(42)

    reference_data = pd.DataFrame({
        "Engine rpm": np.random.normal(700, 100, 1000),
    })

    current_data = pd.DataFrame({
        "Engine rpm": np.random.normal(700, 100, 100),
    })

    with pytest.raises(
        ValueError,
        match="Not enough production samples",
    ):
        detect_drift(
            reference_data,
            current_data,
            min_samples=500,
        )

def test_detect_drift_with_feature_mismatch():
    np.random.seed(42)

    reference_data = pd.DataFrame({
        "Engine rpm": np.random.normal(700, 100, 1000),
        "Coolant temp": np.random.normal(80, 5, 1000),
    })

    current_data = pd.DataFrame({
        "Engine rpm": np.random.normal(700, 100, 500),
        "Wrong feature": np.random.normal(80, 5, 500),
    })

    with pytest.raises(
        ValueError,
        match="Feature mismatch",
    ):
        detect_drift(
            reference_data,
            current_data,
            min_samples=500,
        )

def test_detect_drift_with_normal_distribution():
    rng = np.random.default_rng(42)

    reference_values = rng.normal(
        loc=700,
        scale=100,
        size=5000,
    )

    # Sample current traffic directly from the reference population
    current_values = rng.choice(
        reference_values,
        size=500,
        replace=False,
    )

    reference_data = pd.DataFrame({
        "Engine rpm": reference_values,
    })

    current_data = pd.DataFrame({
        "Engine rpm": current_values,
    })

    results = detect_drift(
        reference_data,
        current_data,
        threshold=0.2,
        min_samples=500,
    )

    assert results["Engine rpm"]["drift_detected"] is False
    assert results["Engine rpm"]["psi"] < 0.2

def test_detect_drift_with_shifted_distribution():
    rng = np.random.default_rng(42)

    reference_data = pd.DataFrame({
        "Engine rpm": rng.normal(
            loc=700,
            scale=100,
            size=5000,
        ),
    })

    current_data = pd.DataFrame({
        "Engine rpm": rng.normal(
            loc=1100,
            scale=100,
            size=500,
        ),
    })

    results = detect_drift(
        reference_data,
        current_data,
        threshold=0.2,
        min_samples=500,
    )

    assert results["Engine rpm"]["drift_detected"] is True
    assert results["Engine rpm"]["psi"] >= 0.2

def test_generate_drift_report():
    rng = np.random.default_rng(42)

    reference_data = pd.DataFrame({
        "Engine rpm": rng.normal(
            loc=700,
            scale=100,
            size=5000,
        ),
    })

    current_data = pd.DataFrame({
        "Engine rpm": rng.normal(
            loc=1100,
            scale=100,
            size=500,
        ),
    })

    report = generate_drift_report(
        reference_data,
        current_data,
        threshold=0.2,
        min_samples=500,
    )

    assert report["monitoring_status"] == "DRIFT_DETECTED"
    assert report["sample_count"] == 500
    assert report["psi_threshold"] == 0.2

    assert "Engine rpm" in report["drifted_features"]

    assert "timestamp" in report
    assert "feature_results" in report

    assert (
        report["feature_results"]["Engine rpm"]["drift_detected"]
        is True
    )

def test_save_drift_report(tmp_path):
    report = {
        "timestamp": "2026-09-07T00:00:00+00:00",
        "monitoring_status": "DRIFT_DETECTED",
        "sample_count": 500,
        "psi_threshold": 0.2,
        "drifted_features": ["Engine rpm"],
        "feature_results": {
            "Engine rpm": {
                "psi": 1.5,
                "drift_detected": True,
            }
        },
    }

    report_file = save_drift_report(
        report,
        output_dir=tmp_path,
    )

    assert report_file.exists()
    assert report_file.suffix == ".json"