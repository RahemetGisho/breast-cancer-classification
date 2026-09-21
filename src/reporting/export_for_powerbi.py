"""Export data tables for a Power BI results dashboard.

Power BI is a reporting/exploration tool over data at rest -- it isn't
built to call a live REST API per user click the way the FastAPI service
is. So rather than wiring Power BI into /predict, this module produces
the fact tables a performance dashboard actually needs: per-test-case
predictions (for a confusion matrix and threshold exploration), global
SHAP feature importance, a one-row metrics summary, and the parsed
production prediction log (for a monitoring view).

Run with: python -m src.reporting.export_for_powerbi
Output: reports/powerbi/*.csv
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.explainability.shap_explain import compute_shap_values
from src.features.preprocessing import split_features_target
from src.models.evaluate import compute_metrics
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


def _load_metadata(config: dict) -> dict:
    registry_dir = resolve_path(config["model_registry"]["dir"])
    metadata_path = registry_dir / config["model_registry"]["metadata_filename"]
    with open(metadata_path) as f:
        return json.load(f)


def export_test_predictions(config: dict, pipeline, output_dir) -> pd.DataFrame:
    """Recreate the exact same train/test split train.py used (same seed,
    same stratify setting), then export one row per test case: every
    original feature, the true label, the predicted label, the predicted
    probability, and a correct/incorrect flag. This is the fact table
    Power BI needs for a confusion matrix, a recall-vs-threshold slider,
    or a "show me the misclassified cases" table.
    """
    df = load_validated_dataframe()
    X, y = split_features_target(df, TARGET_COLUMN)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_seed"],
        stratify=y if config["data"]["stratify"] else None,
    )

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    export_df = X_test.copy()
    export_df.insert(0, "prediction_id", range(len(export_df)))
    export_df["actual_label"] = y_test.map({0: "benign", 1: "malignant"}).values
    export_df["predicted_label"] = pd.Series(y_pred, index=X_test.index).map(
        {0: "benign", 1: "malignant"}
    )
    export_df["probability_malignant"] = y_proba
    export_df["correct"] = export_df["actual_label"] == export_df["predicted_label"]
    export_df["outcome"] = np.select(
        [
            (y_test.values == 1) & (y_pred == 1),
            (y_test.values == 1) & (y_pred == 0),
            (y_test.values == 0) & (y_pred == 0),
            (y_test.values == 0) & (y_pred == 1),
        ],
        ["true_positive", "false_negative", "true_negative", "false_positive"],
        default="unknown",
    )

    path = output_dir / "test_predictions.csv"
    export_df.to_csv(path, index=False)
    logger.info("Exported %d test predictions to %s", len(export_df), path)
    return export_df


def export_model_metrics(metadata: dict, output_dir) -> None:
    """A single-row (or single-row-per-model-version, if this is ever
    run repeatedly and appended) metrics summary -- the KPI-card source
    for the dashboard's overview page.
    """
    metrics = metadata["metrics"]
    row = {
        "model_name": metadata["model_name"],
        "trained_at_utc": metadata["trained_at_utc"],
        "tuned": metadata.get("tuned", False),
        **metrics,
    }
    path = output_dir / "model_metrics.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    logger.info("Exported model metrics summary to %s", path)


def export_shap_feature_importance(
    pipeline, output_dir, sample_size: int = 150
) -> None:
    """Global feature importance: mean absolute SHAP value per feature,
    ranked -- the source table for a horizontal bar chart on the
    dashboard's feature-importance page.
    """
    df = load_validated_dataframe()
    X, _ = split_features_target(df, TARGET_COLUMN)
    sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    shap_values, feature_names = compute_shap_values(pipeline, sample, sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap_value": mean_abs_shap}
    ).sort_values("mean_abs_shap_value", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(path, index=False)
    logger.info("Exported SHAP feature importance to %s", path)


def export_prediction_log(config: dict, output_dir) -> None:
    """Flatten logs/predictions.jsonl (the API's own request/response log)
    into two CSVs: a wide fact table (one row per prediction) and a long
    detail table (one row per prediction x SHAP contributor), joined by
    prediction_id. Splitting it this way -- rather than embedding the
    nested contributor list as a single stringified column -- keeps both
    tables directly importable into Power BI without a manual JSON-parse
    step in Power Query.

    Note: as of this export, this log contains only the handful of
    requests made while testing the API locally, not real production
    traffic. The dashboard page built from it should be labeled as such
    rather than implied to be live production monitoring.
    """
    log_path = resolve_path(config["logging"]["predictions_log_path"])
    if not log_path.exists():
        logger.warning(
            "No prediction log found at %s -- skipping this export.", log_path
        )
        return

    fact_rows = []
    contributor_rows = []
    with open(log_path) as f:
        for prediction_id, line in enumerate(f):
            entry = json.loads(line)
            response = entry["response"]
            contributors = response.pop("top_contributors", [])

            fact_rows.append(
                {
                    "prediction_id": prediction_id,
                    "timestamp_utc": entry["timestamp_utc"],
                    **entry["request"],
                    **response,
                }
            )
            for rank, contributor in enumerate(contributors, start=1):
                contributor_rows.append(
                    {"prediction_id": prediction_id, "rank": rank, **contributor}
                )

    if not fact_rows:
        logger.warning(
            "Prediction log at %s is empty -- skipping this export.", log_path
        )
        return

    pd.DataFrame(fact_rows).to_csv(output_dir / "prediction_log.csv", index=False)
    pd.DataFrame(contributor_rows).to_csv(
        output_dir / "prediction_log_contributors.csv", index=False
    )
    logger.info(
        "Exported %d logged predictions (+ %d contributor rows) to %s",
        len(fact_rows),
        len(contributor_rows),
        output_dir,
    )


def run_export() -> None:
    config = load_config()
    output_dir = resolve_path("reports/powerbi")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _load_registered_pipeline(config)
    metadata = _load_metadata(config)

    test_predictions = export_test_predictions(config, pipeline, output_dir)
    export_model_metrics(metadata, output_dir)
    export_shap_feature_importance(pipeline, output_dir)
    export_prediction_log(config, output_dir)

    # Sanity check: metrics recomputed from the exported test_predictions
    # table should match what's registered in model_metadata.json --
    # if they don't, the export itself has a bug worth catching here
    # rather than a stakeholder noticing a mismatched dashboard number.
    y_true = (test_predictions["actual_label"] == "malignant").astype(int).to_numpy()
    y_pred = (test_predictions["predicted_label"] == "malignant").astype(int).to_numpy()
    y_proba = test_predictions["probability_malignant"].to_numpy()
    recomputed = compute_metrics(y_true, y_pred, y_proba)
    registered = metadata["metrics"]
    assert (
        abs(recomputed["recall_malignant"] - registered["recall_malignant"]) < 1e-9
    ), (
        "Exported test_predictions.csv does not reproduce the registered model's "
        "recall -- export and registry have drifted apart."
    )
    logger.info("Export verified consistent with model_registry/model_metadata.json")


if __name__ == "__main__":
    run_export()
