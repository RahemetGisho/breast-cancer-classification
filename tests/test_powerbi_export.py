"""Tests for src/reporting/export_for_powerbi.py.

Focus: the exported test_predictions.csv must reproduce the exact same
metrics as the registered model's metadata (the export's own internal
consistency check in run_export() proves this at run time, but that
function also has assert statements a test won't catch if someone
weakens them later -- these tests re-verify the same property
independently), and no exported table should contain a nested/stringified
object that would break a Power BI Power Query import.
"""

import pandas as pd
import pytest

from src.models.evaluate import compute_metrics
from src.reporting.export_for_powerbi import (
    export_model_metrics,
    export_prediction_log,
    export_shap_feature_importance,
    export_test_predictions,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def pipeline(config):
    from src.reporting.export_for_powerbi import _load_registered_pipeline

    return _load_registered_pipeline(config)


@pytest.fixture(scope="module")
def metadata(config):
    from src.reporting.export_for_powerbi import _load_metadata

    return _load_metadata(config)


def test_export_test_predictions_reproduces_registered_metrics(
    config, pipeline, metadata, tmp_path
):
    df = export_test_predictions(config, pipeline, tmp_path)

    y_true = (df["actual_label"] == "malignant").astype(int).to_numpy()
    y_pred = (df["predicted_label"] == "malignant").astype(int).to_numpy()
    y_proba = df["probability_malignant"].to_numpy()

    recomputed = compute_metrics(y_true, y_pred, y_proba)
    registered = metadata["metrics"]

    assert recomputed["recall_malignant"] == pytest.approx(
        registered["recall_malignant"]
    )
    assert recomputed["roc_auc"] == pytest.approx(registered["roc_auc"])


def test_export_test_predictions_has_no_nested_objects(config, pipeline, tmp_path):
    """Every column must be a flat scalar -- a list/dict column would
    serialize as a Python repr string in CSV, which Power Query cannot
    parse without a manual JSON-decode step.
    """
    df = export_test_predictions(config, pipeline, tmp_path)
    for col in df.columns:
        sample_value = df[col].iloc[0]
        assert not isinstance(sample_value, (list, dict))


def test_export_model_metrics_writes_single_row(metadata, tmp_path):
    export_model_metrics(metadata, tmp_path)
    result = pd.read_csv(tmp_path / "model_metrics.csv")
    assert len(result) == 1
    assert result.iloc[0]["model_name"] == metadata["model_name"]


def test_export_shap_feature_importance_is_ranked(pipeline, tmp_path):
    export_shap_feature_importance(pipeline, tmp_path, sample_size=30)
    result = pd.read_csv(tmp_path / "shap_feature_importance.csv")

    assert len(result) == 30  # one row per feature (WDBC has 30 features)
    assert list(result["rank"]) == sorted(result["rank"])
    # ranked descending by importance -- rank 1 should have the highest value
    assert result.iloc[0]["mean_abs_shap_value"] == result["mean_abs_shap_value"].max()


def test_export_prediction_log_splits_contributors_into_own_table(config, tmp_path):
    """The regression this guards against: contributors were originally
    embedded as a stringified Python list inside the wide table, which
    Power BI's importer cannot parse without extra manual steps.
    """
    export_prediction_log(config, tmp_path)

    fact_path = tmp_path / "prediction_log.csv"
    contributors_path = tmp_path / "prediction_log_contributors.csv"

    if not fact_path.exists():
        pytest.skip("No prediction log present in this environment to export from.")

    fact_df = pd.read_csv(fact_path)
    assert "top_contributors" not in fact_df.columns

    contributors_df = pd.read_csv(contributors_path)
    assert set(contributors_df.columns) == {
        "prediction_id",
        "rank",
        "feature",
        "shap_value",
        "direction",
    }
    # every contributor row must reference a prediction_id that exists
    # in the fact table -- otherwise the two tables can't be joined
    assert set(contributors_df["prediction_id"]).issubset(set(fact_df["prediction_id"]))
    # shap_value must have parsed as a real float column, not gotten
    # stuck as a string representation of a larger structure
    assert pd.api.types.is_float_dtype(contributors_df["shap_value"])
