"""Load the registered model once at API startup and hold it in memory.

Loading from disk on every request would be both slow and a needless
source of race conditions if the model file is being updated by a
retraining job; loading once at startup and holding a reference is the
standard pattern for model-serving APIs.
"""

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.explainability.shap_explain import (
    compute_shap_values,
    top_contributors_for_instance,
)
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelBundle:
    """Holds the loaded pipeline, its metadata, and a SHAP background
    sample used for explaining individual predictions.
    """

    def __init__(self):
        config = load_config()
        registry_dir = resolve_path(config["model_registry"]["dir"])

        model_path = registry_dir / config["model_registry"]["active_model_filename"]
        metadata_path = registry_dir / config["model_registry"]["metadata_filename"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python -m src.models.train` (and optionally "
                "`python -m src.models.tune`) before starting the API."
            )

        self.pipeline = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Small background sample for SHAP, computed once at startup —
        # recomputing this per-request would be wasteful and slow.
        from src.data.load_data import load_validated_dataframe
        from src.data.schema import TARGET_COLUMN
        from src.features.preprocessing import split_features_target

        df = load_validated_dataframe()
        X, _ = split_features_target(df, TARGET_COLUMN)
        self.background = X.sample(n=min(100, len(X)), random_state=42)
        self.feature_columns = list(X.columns)

        logger.info(
            "Loaded model %s (trained_at=%s) for serving",
            self.metadata.get("model_name"),
            self.metadata.get("trained_at_utc"),
        )

    def predict_one(self, features: dict) -> dict:
        X = pd.DataFrame([features])[self.feature_columns]

        prediction = int(self.pipeline.predict(X)[0])
        probability_malignant = float(self.pipeline.predict_proba(X)[0, 1])

        shap_values, feature_names = compute_shap_values(
            self.pipeline, X, self.background
        )
        top_contributors = top_contributors_for_instance(
            shap_values[0], feature_names, top_n=5
        )

        return {
            "prediction": prediction,
            "prediction_label": "malignant" if prediction == 1 else "benign",
            "probability_malignant": probability_malignant,
            "top_contributors": top_contributors,
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_version_trained_at": self.metadata.get("trained_at_utc", "unknown"),
        }


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Cached singleton accessor — the model loads once per process."""
    return ModelBundle()
