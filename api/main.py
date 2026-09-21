"""FastAPI serving layer for the breast cancer classifier.

Run locally: uvicorn api.main:app --reload
Run via Docker: docker compose up

Every prediction request/response is appended to a JSONL log
(logs/predictions.jsonl) — this is the raw material the drift-check
script (src/utils/drift_check.py) reads to compare incoming feature
distributions against the training distribution.
"""

import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from api.model_loader import get_model_bundle
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "Serves malignant/benign predictions from FNA cell measurements, "
        "with a SHAP-based explanation attached to every prediction. "
        "This is a decision-support tool, not a diagnostic replacement;  "
        "predictions should be reviewed by a qualified clinician."
    ),
    version="1.0.0",
)


def _log_prediction(request_dict: dict, response_dict: dict) -> None:
    config = load_config()
    log_path = resolve_path(config["logging"]["predictions_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request": request_dict,
        "response": response_dict,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        bundle = get_model_bundle()
        return HealthResponse(
            status="ok", model_loaded=True, model_name=bundle.metadata.get("model_name")
        )
    except FileNotFoundError:
        return HealthResponse(status="model_not_found", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    request_dict = request.model_dump()

    try:
        result = bundle.predict_one(request_dict)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 500, but log details
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    _log_prediction(request_dict, result)

    return PredictionResponse(**result)
