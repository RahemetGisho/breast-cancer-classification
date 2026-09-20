"""Hyperparameter tuning with Optuna.

Rather than tuning all three candidate models (expensive, and the losers
from the initial comparison rarely catch up), this tunes the model that
won Task 5's comparison — read from the registry's current metadata —
using a Bayesian search (Optuna's TPE sampler) instead of grid/random
search. Objective: mean cross-validated recall on the malignant class,
which is the metric that matters clinically.

Run with: python -m src.models.tune
"""

import json
import time
from datetime import datetime, timezone

import joblib
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.features.preprocessing import build_pipeline, split_features_target
from src.models.evaluate import compute_metrics, format_metrics_report
from src.models.model_factory import build_model
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_hyperparams(trial: optuna.Trial, model_name: str) -> dict:
    """Search space per model type. Kept intentionally modest in range —
    a search space this small doesn't need hundreds of trials to converge,
    and an overly wide space on this small a dataset just invites overfit
    hyperparameters that look good on paper and don't generalize.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    raise ValueError(f"No search space defined for model: {model_name!r}")


def tune_model(config: dict | None = None) -> dict:
    config = config or load_config()
    seed = config["project"]["random_seed"]
    imbalance_strategy = config["preprocessing"]["imbalance_strategy"]
    scaler_name = config["preprocessing"]["scaler"]
    cv_folds = config["training"]["cv_folds"]
    n_trials = config["tuning"]["n_trials"]
    timeout = config["tuning"]["timeout_seconds"]

    registry_dir = resolve_path(config["model_registry"]["dir"])
    metadata_path = registry_dir / config["model_registry"]["metadata_filename"]
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existing model metadata at {metadata_path}. Run src.models.train first "
            "so tuning knows which model type won the initial comparison."
        )
    with open(metadata_path) as f:
        current_metadata = json.load(f)
    model_name = current_metadata["model_name"]
    logger.info("Tuning model: %s (winner of initial comparison)", model_name)

    df = load_validated_dataframe()
    X, y = split_features_target(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=y if config["data"]["stratify"] else None,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = _suggest_hyperparams(trial, model_name)
        model = build_model(
            model_name, imbalance_strategy, random_seed=seed, **hyperparams
        )
        pipeline = build_pipeline(model, scaler_name, imbalance_strategy, seed)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.time() - start

    logger.info(
        "Tuning finished: %d trials in %.1fs, best CV recall=%.4f",
        len(study.trials),
        elapsed,
        study.best_value,
    )

    # Refit the tuned model on the full training set and evaluate on the
    # untouched held-out test set — the same test set train.py used, so
    # results are directly comparable to the untuned baseline.
    best_model = build_model(
        model_name, imbalance_strategy, random_seed=seed, **study.best_params
    )
    tuned_pipeline = build_pipeline(best_model, scaler_name, imbalance_strategy, seed)
    tuned_pipeline.fit(X_train, y_train)

    y_pred = tuned_pipeline.predict(X_test)
    y_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
    tuned_metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
    logger.info(format_metrics_report(f"{model_name} (tuned)", tuned_metrics))

    baseline_recall = current_metadata["metrics"]["recall_malignant"]
    tuned_recall = tuned_metrics["recall_malignant"]
    improved = tuned_recall >= baseline_recall

    if improved:
        _save_tuned_model(tuned_pipeline, model_name, study, tuned_metrics, config)
        logger.info(
            "Tuned model adopted: recall %.4f -> %.4f", baseline_recall, tuned_recall
        )
    else:
        logger.info(
            "Tuned model NOT adopted: recall %.4f would regress from baseline %.4f. "
            "Keeping existing registered model.",
            tuned_recall,
            baseline_recall,
        )

    return {
        "model_name": model_name,
        "best_params": study.best_params,
        "cv_recall": study.best_value,
        "test_metrics": tuned_metrics,
        "adopted": improved,
        "n_trials_run": len(study.trials),
    }


def _save_tuned_model(
    pipeline, model_name: str, study, metrics: dict, config: dict
) -> None:
    registry_dir = resolve_path(config["model_registry"]["dir"])
    model_path = registry_dir / config["model_registry"]["active_model_filename"]
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "tuned": True,
        "tuning": {
            "n_trials": len(study.trials),
            "best_params": study.best_params,
            "best_cv_recall": study.best_value,
        },
    }
    metadata_path = registry_dir / config["model_registry"]["metadata_filename"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    tune_model()
