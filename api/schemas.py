"""Pydantic schemas for the prediction API.

Using an explicit model for each of the 30 features (rather than a raw
dict) means malformed requests are rejected by FastAPI's validation
layer before they ever reach the model — the same "fail loudly, fail
early" principle applied at the training data layer now applies at
serving time too.
"""

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.data.schema import FEATURE_COLUMNS

# Built dynamically from the same FEATURE_COLUMNS the data validation
# layer uses, so the API contract and the training data contract can
# never silently drift apart. Each field is a required, non-negative
# float, matching the physical-measurement constraint enforced in
# src/data/schema.py.
PredictionRequest = create_model(
    "PredictionRequest",
    **{
        col: (float, Field(..., ge=0, description=f"{col} measurement"))
        for col in FEATURE_COLUMNS
    },
)


class Contributor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    # Pydantic reserves the "model_" prefix for its own internal namespace
    # (e.g. model_config, model_dump) and warns when a field also starts
    # with it. model_name / model_version_trained_at are genuinely ours
    # (which model served this prediction, and when it was trained), so
    # we silence the warning rather than rename fields the API consumer
    # already sees in a less clear way.
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0 = benign, 1 = malignant")
    prediction_label: str
    probability_malignant: float
    top_contributors: list[Contributor]
    model_name: str
    model_version_trained_at: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
