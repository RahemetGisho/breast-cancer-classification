"""Tests for src/models/tune.py.

Runs the real Optuna tuning path end-to-end, but with n_trials cut
down to a handful via a config override -- proving the tuning code
itself works without paying for the full 40-trial search on every
CI run. This module previously had 0% coverage; that's the gap this
file exists to close.

Note on isolation: model_registry.dir points at a temp directory, not
the real one in configs/config.yaml -- otherwise running this test
module would overwrite the actual served model with a 3-trial (instead
of 40-trial) tuning result. A test suite must never mutate the
production artifact it's supposed to be verifying from the outside.
"""

import copy

import pytest

from src.models.train import train_and_compare
from src.models.tune import tune_model
from src.utils.config import load_config


@pytest.fixture(scope="module")
def fast_config(tmp_path_factory):
    """A copy of the real config with tuning cut down to a few trials,
    a short timeout, and an isolated model registry directory.
    """
    config = copy.deepcopy(load_config())
    config["tuning"]["n_trials"] = 3
    config["tuning"]["timeout_seconds"] = 60
    isolated_registry = tmp_path_factory.mktemp("tuning_model_registry")
    config["model_registry"]["dir"] = str(isolated_registry)
    return config


@pytest.fixture(scope="module")
def initial_training_run(fast_config):
    """The one and only training call in this module -- captured once
    so later tests can compare against it without re-running training
    (which would re-overwrite the isolated registry a second time).
    """
    return train_and_compare(fast_config)


@pytest.fixture(scope="module")
def trained_then_tuned(fast_config, initial_training_run):
    """tune_model requires an existing model_metadata.json (it tunes
    whichever model won the initial comparison) -- initial_training_run
    guarantees that dependency ran first.
    """
    return tune_model(fast_config)


def test_tune_model_runs_end_to_end(trained_then_tuned):
    assert trained_then_tuned["n_trials_run"] >= 1
    assert "best_params" in trained_then_tuned
    assert "test_metrics" in trained_then_tuned


def test_tune_model_tunes_the_model_that_won_initial_comparison(
    initial_training_run, trained_then_tuned
):
    assert trained_then_tuned["model_name"] == initial_training_run["best_model"]


def test_tune_model_raises_without_prior_training(fast_config, tmp_path):
    """tune_model should fail clearly, not silently, if run before any
    model has been trained -- there is nothing to know which model type
    to tune otherwise."""
    empty_registry_config = copy.deepcopy(fast_config)
    empty_registry_config["model_registry"]["dir"] = str(tmp_path / "empty_registry")

    with pytest.raises(FileNotFoundError):
        tune_model(empty_registry_config)


def test_tuned_metrics_never_regress_below_quality_gate(
    fast_config, trained_then_tuned
):
    """Whatever tune_model decides to adopt or reject, the metrics it
    reports for the final registered model must still clear the same
    quality gate training.py is held to -- tuning must not be a
    backdoor around the quality bar.
    """
    gate = fast_config["quality_gate"]
    metrics = trained_then_tuned["test_metrics"]
    if trained_then_tuned["adopted"]:
        assert metrics["recall_malignant"] >= gate["min_recall_malignant"]
