from pathlib import Path
import sys

from fastapi import FastAPI, Response
from pydantic import BaseModel


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


# Load artifacts once when the API starts
model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)


app = FastAPI(
    title="Predictive Maintenance API",
    description="API for engine condition classification.",
    version="1.0.0",
)


class EngineData(BaseModel):
    engine_rpm: float
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float


@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": metadata["model_version"],
    }


@app.post("/predict")
def predict_engine_condition(data: EngineData):

    input_data = {
        "Engine rpm": data.engine_rpm,
        "Lub oil pressure": data.lub_oil_pressure,
        "Fuel pressure": data.fuel_pressure,
        "Coolant pressure": data.coolant_pressure,
        "lub oil temp": data.lub_oil_temp,
        "Coolant temp": data.coolant_temp,
    }

    result = predict_single(
        model,
        metadata,
        input_data,
    )

    return result


# SageMaker health-check endpoint
@app.get("/ping")
def ping():
    return Response(status_code=200)


# SageMaker inference endpoint
@app.post("/invocations")
def invocations(data: EngineData):

    input_data = {
        "Engine rpm": data.engine_rpm,
        "Lub oil pressure": data.lub_oil_pressure,
        "Fuel pressure": data.fuel_pressure,
        "Coolant pressure": data.coolant_pressure,
        "lub oil temp": data.lub_oil_temp,
        "Coolant temp": data.coolant_temp,
    }

    result = predict_single(
        model,
        metadata,
        input_data,
    )

    return result