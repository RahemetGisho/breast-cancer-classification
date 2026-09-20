"""Tests for the training pipeline and the model-quality gate.

The quality-gate test is the important one here: it is not a code test,
it is a *model* test. It fails CI if a retrained model regresses below
the clinically-justified minimum bar defined in configs/config.yaml,
independent of whether the code technically runs without error. A
pipeline that runs cleanly but produces a worse model is still a
failure for this project.

Note on isolation: these tests point model_registry.dir at a temp
directory rather than the real configs/config.yaml path. Without this,
running the test suite would silently overwrite the actual served
model in models/model_registry/ with whatever this test run produced
-- a test suite must never mutate the production artifact it's
supposed to be verifying from the outside.
"""

import copy
import json

import pytest

from src.models.model_factory import build_model
from src.models.train import train_and_compare
from src.utils.config import load_config, resolve_path


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    cfg = copy.deepcopy(load_config())
    isolated_registry = tmp_path_factory.mktemp("model_registry")
    cfg["model_registry"]["dir"] = str(isolated_registry)
    return cfg


@pytest.fixture(scope="module")
def training_run(config):
    """Run the real training pipeline once per test module and share the
    result across tests -- retraining per-test would be wasteful and
    the pipeline is deterministic given the fixed seed anyway.
    """
    return train_and_compare(config)


def test_training_runs_end_to_end_without_error(training_run):
    assert "results" in training_run
    assert "best_model" in training_run
    assert training_run["best_model"] in training_run["results"]


def test_all_configured_models_were_trained(config, training_run):
    for model_name in config["training"]["models"]:
        assert model_name in training_run["results"]
        metrics = training_run["results"][model_name]["metrics"]
        # every reported metric key must be present and a real number
        for key in ("accuracy", "recall_malignant", "precision_malignant", "roc_auc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


def test_best_model_selected_by_configured_scoring_metric(config, training_run):
    scoring_metric = config["training"]["scoring_metric"]
    scores = {
        name: result["metrics"][f"{scoring_metric}_malignant"]
        for name, result in training_run["results"].items()
    }
    assert training_run["best_model"] == max(scores, key=scores.get)


def test_model_registry_artifacts_are_saved(config, training_run):
    registry_dir = resolve_path(config["model_registry"]["dir"])
    model_path = registry_dir / config["model_registry"]["active_model_filename"]
    metadata_path = registry_dir / config["model_registry"]["metadata_filename"]

    assert model_path.exists()
    assert metadata_path.exists()

    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["model_name"] == training_run["best_model"]
    assert "metrics" in metadata


def test_model_quality_gate(config, training_run):
    """The actual production gate: the registered model must clear the
    minimum recall and ROC-AUC bar defined in config, or this test (and
    CI) fails. This is what stops a worse model from silently being
    shipped.
    """
    best_metrics = training_run["results"][training_run["best_model"]]["metrics"]
    gate = config["quality_gate"]

    assert best_metrics["recall_malignant"] >= gate["min_recall_malignant"], (
        f"Model recall {best_metrics['recall_malignant']:.4f} fell below the "
        f"required minimum {gate['min_recall_malignant']} -- this model must "
        "not be shipped."
    )
    assert best_metrics["roc_auc"] >= gate["min_roc_auc"], (
        f"Model ROC-AUC {best_metrics['roc_auc']:.4f} fell below the required "
        f"minimum {gate['min_roc_auc']}."
    )


def test_model_factory_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        build_model("not_a_real_model", imbalance_strategy="class_weight")


def test_model_factory_applies_class_weight_when_configured():
    model = build_model("logistic_regression", imbalance_strategy="class_weight")
    assert model.class_weight == "balanced"


def test_model_factory_omits_class_weight_for_smote():
    """When SMOTE already balances the training data, the estimator
    should NOT also apply class_weight='balanced' -- doing both would
    double-correct for imbalance and bias the model the other way."""
    model = build_model("logistic_regression", imbalance_strategy="smote")
    assert model.class_weight is None
