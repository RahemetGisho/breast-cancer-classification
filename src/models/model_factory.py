"""Factory for candidate model estimators.

Kept separate from train.py so tuning code (Optuna) and training code can
both construct a fresh, correctly-configured estimator without duplicating
the class_weight / random_state wiring.
"""

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(
    model_name: str,
    imbalance_strategy: str,
    random_seed: int = 42,
    **hyperparams,
) -> BaseEstimator:
    """Construct a model instance.

    class_weight="balanced" is only applied when imbalance_strategy is
    "class_weight" — if SMOTE already balanced the training data, adding
    class weighting on top would double-correct for imbalance.
    """
    use_class_weight = imbalance_strategy == "class_weight"

    if model_name == "logistic_regression":
        # solver defaults to "saga" (not sklearn's default "lbfgs") because
        # it's the only solver supporting both l1 and l2 penalties, which
        # matters for tuning: Optuna's study.best_params only contains keys
        # obtained via trial.suggest_*, so a fixed non-suggested override
        # passed into build_model at objective-eval time would silently
        # vanish on refit. Setting it here means every code path
        # (untuned training, tuning objective, tuned refit) gets a solver
        # consistent with whichever penalty ends up chosen.
        params = dict(max_iter=5000, random_state=random_seed, solver="saga")
        if use_class_weight:
            params["class_weight"] = "balanced"
        params.update(hyperparams)
        return LogisticRegression(**params)

    if model_name == "random_forest":
        params = dict(random_state=random_seed, n_jobs=-1)
        if use_class_weight:
            params["class_weight"] = "balanced"
        params.update(hyperparams)
        return RandomForestClassifier(**params)

    if model_name == "xgboost":
        params = dict(random_state=random_seed, eval_metric="logloss")
        if use_class_weight:
            # XGBoost has no class_weight param; scale_pos_weight is its
            # equivalent lever for imbalanced binary classification.
            params.setdefault(
                "scale_pos_weight", hyperparams.pop("scale_pos_weight", 1.0)
            )
        params.update(hyperparams)
        return XGBClassifier(**params)

    raise ValueError(f"Unknown model_name: {model_name!r}")
