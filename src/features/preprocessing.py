"""Preprocessing pipeline construction.

Two design decisions matter here:

1. Scaling lives inside the sklearn Pipeline with the model, not as a
   separate fit_transform step. This means the exact same scaler fitted
   on training data is what runs at inference time — there is no way to
   "forget" to scale an incoming request, because it isn't a separate
   step at all.

2. Imbalance handling is a configurable strategy, chosen and justified
   in REPORT.md rather than defaulted silently:
   - "class_weight": no resampling, just reweight the loss. Safe by
     construction (can never leak information across the train/test
     split) — this is the default.
   - "smote": oversample the minority class. Must only ever be applied
     to the training fold, never before the split, or it leaks synthetic
     neighbors of test points into training.
"""

from typing import Literal

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ImbalanceStrategy = Literal["class_weight", "smote"]


def get_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {scaler_name!r}")


def build_pipeline(
    model: BaseEstimator,
    scaler_name: str = "standard",
    imbalance_strategy: ImbalanceStrategy = "class_weight",
    random_seed: int = 42,
) -> ImbPipeline:
    """Build a single Pipeline containing scaling, (optional) resampling,
    and the model. This is the object that gets fit, cross-validated,
    serialized, and served — always as one unit.
    """
    steps = [("scaler", get_scaler(scaler_name))]

    if imbalance_strategy == "smote":
        steps.append(("smote", SMOTE(random_state=random_seed)))
    elif imbalance_strategy != "class_weight":
        raise ValueError(f"Unknown imbalance_strategy: {imbalance_strategy!r}")
    # "class_weight" requires no pipeline step — it's handled by passing
    # class_weight="balanced" to the estimator itself at construction time.

    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def split_features_target(
    df: pd.DataFrame, target_column: str = "target"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
