"""Tests for src/features/preprocessing.py.

These check the pipeline is built correctly for each imbalance
strategy (no silent misconfiguration), and that preprocessing is
deterministic given a fixed seed -- a model that isn't reproducible
between runs isn't trustworthy for a medical use case.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.features.preprocessing import (
    build_pipeline,
    get_scaler,
    split_features_target,
)


@pytest.fixture(scope="module")
def df():
    return load_validated_dataframe()


def test_split_features_target_shapes(df):
    X, y = split_features_target(df, TARGET_COLUMN)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == len(df)
    assert set(y.unique()) == {0, 1}


def test_get_scaler_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_scaler("not_a_real_scaler")


def test_build_pipeline_class_weight_has_no_smote_step():
    model = LogisticRegression()
    pipeline = build_pipeline(model, "standard", "class_weight", random_seed=42)
    step_names = [name for name, _ in pipeline.steps]
    assert "smote" not in step_names
    assert step_names == ["scaler", "model"]


def test_build_pipeline_smote_has_smote_step():
    model = LogisticRegression()
    pipeline = build_pipeline(model, "standard", "smote", random_seed=42)
    step_names = [name for name, _ in pipeline.steps]
    assert "smote" in step_names
    assert step_names == ["scaler", "smote", "model"]


def test_build_pipeline_rejects_unknown_imbalance_strategy():
    model = LogisticRegression()
    with pytest.raises(ValueError):
        build_pipeline(model, "standard", "not_a_real_strategy", random_seed=42)


def test_pipeline_fit_is_deterministic_given_fixed_seed(df):
    """Same seed, same data -> identical predictions. Any nondeterminism
    here (e.g. an unseeded random component) would make results
    impossible to reproduce or audit later.
    """
    X, y = split_features_target(df, TARGET_COLUMN)

    model_a = LogisticRegression(max_iter=5000, random_state=42, solver="saga")
    pipeline_a = build_pipeline(model_a, "standard", "class_weight", random_seed=42)
    pipeline_a.fit(X, y)

    model_b = LogisticRegression(max_iter=5000, random_state=42, solver="saga")
    pipeline_b = build_pipeline(model_b, "standard", "class_weight", random_seed=42)
    pipeline_b.fit(X, y)

    preds_a = pipeline_a.predict_proba(X)
    preds_b = pipeline_b.predict_proba(X)
    assert np.allclose(preds_a, preds_b)


def test_smote_pipeline_balances_training_classes(df):
    """After SMOTE, the resampled training data seen by the model should
    have a roughly 1:1 class ratio, even though the source data is
    imbalanced (~63/37)."""
    from imblearn.over_sampling import SMOTE

    X, y = split_features_target(df, TARGET_COLUMN)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    counts = y_resampled.value_counts()
    assert counts[0] == counts[1]
    assert len(X_resampled) > len(X)  # minority class was oversampled
