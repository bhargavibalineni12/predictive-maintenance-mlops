from fastapi.testclient import TestClient

from app.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "model_version" in data

def test_predict_endpoint():
    payload = {
        "engine_rpm": 700,
        "lub_oil_pressure": 2.493592,
        "fuel_pressure": 11.790927,
        "coolant_pressure": 3.178981,
        "lub_oil_temp": 84.144163,
        "coolant_temp": 81.632187,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "threshold" in data

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["threshold"] <= 1.0

def test_predict_endpoint_missing_field():
    payload = {
        "engine_rpm": 700,
        "lub_oil_pressure": 2.493592,
        "fuel_pressure": 11.790927,
        "coolant_pressure": 3.178981,
        "lub_oil_temp": 84.144163,
        # coolant_temp intentionally missing
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422