"""Cached data and model access for the Streamlit dashboard.

Every function here either loads something from disk (the Power BI
exports, the registered model) or performs an on-demand SHAP
computation -- always reusing the same src/ modules the API and the
export script already use, rather than reimplementing any of that
logic here. The dashboard's own job stays thin: load, cache, display.

st.cache_data is for plain data (DataFrames, dicts) -- Streamlit hashes
the return value and skips recomputation on rerun. st.cache_resource is
for objects that shouldn't be copied/hashed the normal way (a fitted
sklearn pipeline) -- Streamlit keeps exactly one shared instance alive
across reruns and sessions, which matters here for the same reason
Module 10's @lru_cache singleton mattered in the API: loading the
model and its background sample is real work you don't want repeated
on every button click.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import FEATURE_COLUMNS  # noqa: E402
from src.explainability.shap_explain import compute_shap_values  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

POWERBI_DIR = PROJECT_ROOT / "reports" / "powerbi"


@st.cache_data
def get_config() -> dict:
    return load_config()


@st.cache_data
def load_model_metrics() -> dict:
    df = pd.read_csv(POWERBI_DIR / "model_metrics.csv")
    return df.iloc[0].to_dict()


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(POWERBI_DIR / "test_predictions.csv")


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(POWERBI_DIR / "shap_feature_importance.csv")


@st.cache_data
def load_prediction_log() -> pd.DataFrame | None:
    path = POWERBI_DIR / "prediction_log.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_prediction_log_contributors() -> pd.DataFrame | None:
    path = POWERBI_DIR / "prediction_log_contributors.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_pipeline_and_metadata():
    config = load_config()
    registry_dir = resolve_path(config["model_registry"]["dir"])
    pipeline = joblib.load(
        registry_dir / config["model_registry"]["active_model_filename"]
    )

    import json

    with open(registry_dir / config["model_registry"]["metadata_filename"]) as f:
        metadata = json.load(f)
    return pipeline, metadata


@st.cache_resource
def get_background_sample() -> pd.DataFrame:
    """A fixed sample of real feature rows, used as SHAP's reference
    point for on-demand local explanations -- same role as
    src/explainability/shap_explain.py's own background sample,
    just sourced from the already-exported test_predictions.csv
    instead of reloading the full training set again.
    """
    test_df = load_test_predictions()
    return test_df[FEATURE_COLUMNS].sample(n=min(100, len(test_df)), random_state=42)


def explain_row(feature_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Compute the top SHAP contributors for one row of raw feature
    values (a 1-row DataFrame with FEATURE_COLUMNS). Reuses the exact
    same compute_shap_values function the API and the export script
    call -- no explainer logic is reimplemented here.
    """
    from src.explainability.shap_explain import top_contributors_for_instance

    pipeline, _ = load_pipeline_and_metadata()
    background = get_background_sample()

    shap_values, feature_names = compute_shap_values(pipeline, feature_row, background)
    return top_contributors_for_instance(shap_values[0], feature_names, top_n=top_n)


def predict_row(feature_row: pd.DataFrame) -> dict:
    """Run one row of raw features through the registered pipeline and
    return prediction + probability + top SHAP contributors -- the
    Streamlit-native equivalent of the API's predict_one(), reusing
    the same underlying pipeline and explainability calls.
    """
    pipeline, metadata = load_pipeline_and_metadata()
    X = feature_row[FEATURE_COLUMNS]

    prediction = int(pipeline.predict(X)[0])
    probability_malignant = float(pipeline.predict_proba(X)[0, 1])
    contributors = explain_row(feature_row)

    return {
        "prediction": prediction,
        "prediction_label": "malignant" if prediction == 1 else "benign",
        "probability_malignant": probability_malignant,
        "top_contributors": contributors,
        "model_name": metadata.get("model_name", "unknown"),
    }
