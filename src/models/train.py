"""Train and compare candidate models.

Every run (per model) is logged to MLflow with its params and metrics.
The best model by the configured scoring_metric on the held-out test set
is serialized to the model registry along with metadata describing why
it was chosen. Run with: python -m src.models.train
"""

import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.features.preprocessing import build_pipeline, split_features_target
from src.models.evaluate import compute_metrics, format_metrics_report
from src.models.model_factory import build_model
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def train_and_compare(config: dict | None = None) -> dict:
    """Train every candidate model, evaluate on a held-out test set, and
    persist the best one. Returns a results dict for programmatic use
    (e.g. by tests or the tuning script).
    """
    config = config or load_config()
    seed = config["project"]["random_seed"]
    imbalance_strategy = config["preprocessing"]["imbalance_strategy"]
    scaler_name = config["preprocessing"]["scaler"]
    scoring_metric = config["training"]["scoring_metric"]
    cv_folds = config["training"]["cv_folds"]

    mlflow_db_path = resolve_path("mlruns.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("breast-cancer-classification")

    df = load_validated_dataframe()
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=y if config["data"]["stratify"] else None,
    )
    logger.info(
        "Train/test split: %d train (%d malignant), %d test (%d malignant)",
        len(X_train),
        int(y_train.sum()),
        len(X_test),
        int(y_test.sum()),
    )

    results = {}
    best_model_name = None
    best_score = -np.inf
    best_pipeline = None

    for model_name in config["training"]["models"]:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "imbalance_strategy": imbalance_strategy,
                    "scaler": scaler_name,
                    "cv_folds": cv_folds,
                }
            )

            model = build_model(model_name, imbalance_strategy, random_seed=seed)
            pipeline = build_pipeline(model, scaler_name, imbalance_strategy, seed)

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1
            )
            mlflow.log_metric("cv_recall_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_recall_std", float(cv_scores.std()))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, float)}
            )

            logger.info(format_metrics_report(model_name, metrics))

            results[model_name] = {
                "metrics": metrics,
                "cv_recall_mean": float(cv_scores.mean()),
                "cv_recall_std": float(cv_scores.std()),
            }

            score = metrics.get(
                f"{scoring_metric}_malignant", metrics.get(scoring_metric)
            )
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline

    _save_best_model(best_pipeline, best_model_name, results[best_model_name], config)

    logger.info("Best model: %s (%s=%.4f)", best_model_name, scoring_metric, best_score)
    return {"results": results, "best_model": best_model_name}


def _save_best_model(
    pipeline, model_name: str, model_result: dict, config: dict
) -> None:
    registry_dir = resolve_path(config["model_registry"]["dir"])
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_path = registry_dir / config["model_registry"]["active_model_filename"]
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": model_result["metrics"],
        "cv_recall_mean": model_result["cv_recall_mean"],
        "cv_recall_std": model_result["cv_recall_std"],
    }
    metadata_path = registry_dir / config["model_registry"]["metadata_filename"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved best model (%s) to %s", model_name, model_path)


if __name__ == "__main__":
    train_and_compare()
