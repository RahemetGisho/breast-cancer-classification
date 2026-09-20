"""SHAP-based explainability for the registered model.

Two outputs:
1. Global: a summary plot showing which features drive predictions
   across the whole dataset -- for REPORT.md and the written analysis.
2. Local: compute_shap_values() + top_contributors_for_instance(), which
   the API calls per-prediction to return the top contributing features
   for that specific patient's result -- what actually matters in a
   clinical context (a doctor needs to know why this patient, not just
   which features matter on average).

Note on the explainer choice: the registered pipeline can be logistic
regression, random forest, or xgboost depending on what won training or
tuning, so we use shap.Explainer's model-agnostic dispatch rather than
hardcoding a tree or linear explainer, and we explain the scaled
feature space the model actually sees (post-pipeline scaling) -- SHAP
values on unscaled inputs would misattribute importance for a model
that was fit on standardized features.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display available in this environment / in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.features.preprocessing import split_features_target
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_registered_pipeline(config: dict):
    registry_dir = resolve_path(config["model_registry"]["dir"])
    model_path = registry_dir / config["model_registry"]["active_model_filename"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"No registered model at {model_path}. Run training first."
        )
    return joblib.load(model_path)


def _scale_inputs(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through every pipeline step except the final model -- i.e.
    apply the same scaling the model was trained on, without invoking
    resampling (SMOTE), which is a training-only step.
    """
    Xt = X
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return pd.DataFrame(Xt, columns=X.columns, index=X.index)


def _build_explainer(pipeline, background_data: pd.DataFrame) -> shap.Explainer:
    """Build a SHAP explainer over the model's scaled input space.

    The model itself was fit inside the pipeline on a plain numpy array
    (the scaler's output), so predict_fn strips the DataFrame back to
    .values before calling into it -- this keeps SHAP's outward-facing
    background_data as a labeled DataFrame (needed for readable plot
    labels) while avoiding sklearn's "fitted without feature names"
    warning that comes from feeding it a named frame at predict time.
    """

    def predict_fn(X):
        X_values = X.values if hasattr(X, "values") else X
        return pipeline.named_steps["model"].predict_proba(X_values)[:, 1]

    return shap.Explainer(predict_fn, background_data)


def compute_shap_values(
    pipeline, X: pd.DataFrame, background: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for X against a background sample.

    Both X and background are scaled internally before being handed to
    the explainer, so callers pass raw (unscaled) feature DataFrames --
    the same shape of data the API receives from a client.

    Returns (shap_values, feature_names) where shap_values has shape
    (n_rows_in_X, n_features) and feature_names gives the column order
    those values correspond to.
    """
    scaled_X = _scale_inputs(pipeline, X)
    scaled_background = _scale_inputs(pipeline, background)

    explainer = _build_explainer(pipeline, scaled_background)
    shap_values = explainer(scaled_X)

    return shap_values.values, list(X.columns)


def top_contributors_for_instance(
    shap_row: np.ndarray, feature_names: list[str], top_n: int = 5
) -> list[dict]:
    """Rank one row of SHAP values by absolute contribution and label
    each with its direction of effect on malignancy risk.

    This is a pure function (no model/data access) so it's cheap to unit
    test in isolation from the explainer machinery above.
    """
    order = np.argsort(-np.abs(shap_row))[:top_n]
    return [
        {
            "feature": feature_names[i],
            "shap_value": float(shap_row[i]),
            "direction": (
                "increases_malignant_risk"
                if shap_row[i] > 0
                else "decreases_malignant_risk"
            ),
        }
        for i in order
    ]


def generate_global_summary(
    config: dict | None = None, output_path: Path | None = None
) -> Path:
    """Produce a SHAP summary plot (global feature importance) over a
    sample of the dataset and save it as a PNG for the report.
    """
    config = config or load_config()
    pipeline = _load_registered_pipeline(config)

    df = load_validated_dataframe()
    X, _ = split_features_target(df, TARGET_COLUMN)

    # A subsample as background/explanation set keeps this fast; SHAP
    # values are stable well below the full 569-row dataset.
    sample = X.sample(n=min(150, len(X)), random_state=42)

    shap_values, feature_names = compute_shap_values(pipeline, sample, sample)

    output_path = output_path or resolve_path("reports/figures/shap_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values, sample.to_numpy(), feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Saved SHAP global summary plot to %s", output_path)
    return output_path


if __name__ == "__main__":
    generate_global_summary()
