import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.log"


logger = logging.getLogger("prediction_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


def log_prediction(
    input_data: dict,
    prediction: int,
    probability: float,
    threshold: float,
    model_version: str,
):
   

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
    }

    logger.info(json.dumps(log_entry))