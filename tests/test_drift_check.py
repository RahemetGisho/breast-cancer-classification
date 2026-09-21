"""Tests for drift_check.py.

Uses synthetic production data rather than a real prediction log so the
test doesn't depend on the API having been run first. Two cases: data
drawn from the same distribution as training data should not be flagged,
and data shifted far outside that distribution should be.
"""

import pytest

from src.data.load_data import load_validated_dataframe
from src.data.schema import TARGET_COLUMN
from src.utils.drift_check import check_drift


@pytest.fixture(scope="module")
def reference_df():
    return load_validated_dataframe().drop(columns=[TARGET_COLUMN])


def test_no_drift_when_production_resembles_training(reference_df):
    # Simulate production data that's just a resample of the training
    # distribution itself — should show no drift.
    production_df = reference_df.sample(n=100, replace=True, random_state=1)

    results = check_drift(production_df, reference_df)

    drifted = [r for r in results if r["drifted"]]
    assert len(drifted) == 0


def test_drift_detected_when_distribution_shifts(reference_df):
    # Shift every feature by several standard deviations — an obvious,
    # unmistakable distribution change.
    shifted_df = reference_df.copy().sample(n=100, replace=True, random_state=2)
    for col in shifted_df.columns:
        shifted_df[col] = shifted_df[col] + reference_df[col].std() * 10

    results = check_drift(shifted_df, reference_df)

    drifted = [r for r in results if r["drifted"]]
    # Every feature was shifted, so every feature should be flagged.
    assert len(drifted) == len(results)


def test_check_drift_returns_one_result_per_feature(reference_df):
    production_df = reference_df.sample(n=50, random_state=3)
    results = check_drift(production_df, reference_df)
    assert len(results) == len(reference_df.columns)
    for r in results:
        assert "p_value" in r
        assert "ks_statistic" in r
        assert isinstance(r["drifted"], bool)


def test_load_production_requests_parses_jsonl_log(tmp_path):
    """load_production_requests reads the same JSONL format api/main.py
    writes -- prove it parses real log lines, not just an idealized dict.
    """
    import json

    from src.utils.drift_check import load_production_requests

    log_path = tmp_path / "predictions.jsonl"
    entries = [
        {
            "timestamp_utc": "2026-01-01T00:00:00",
            "request": {"radius_mean": 14.0},
            "response": {},
        },
        {
            "timestamp_utc": "2026-01-01T00:01:00",
            "request": {"radius_mean": 15.0},
            "response": {},
        },
    ]
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    df = load_production_requests(log_path)
    assert len(df) == 2
    assert list(df["radius_mean"]) == [14.0, 15.0]


def test_load_production_requests_missing_file_raises(tmp_path):
    from src.utils.drift_check import load_production_requests

    with pytest.raises(FileNotFoundError):
        load_production_requests(tmp_path / "does_not_exist.jsonl")


def test_run_drift_check_end_to_end(tmp_path, monkeypatch):
    """Exercise the full run_drift_check() path -- config lookup, log
    loading, and the KS comparison -- against a temp log file, the same
    monkeypatch pattern test_api.py uses for its logging test.
    """
    import json

    from src.data.load_data import load_validated_dataframe
    from src.data.schema import TARGET_COLUMN
    from src.utils import config as config_module
    from src.utils.drift_check import run_drift_check

    df = load_validated_dataframe().drop(columns=[TARGET_COLUMN])
    sample = df.sample(n=40, random_state=7)

    log_path = tmp_path / "predictions.jsonl"
    with open(log_path, "w") as f:
        for _, row in sample.iterrows():
            f.write(json.dumps({"request": row.to_dict(), "response": {}}) + "\n")

    original_load_config = config_module.load_config

    def _patched_load_config(*args, **kwargs):
        cfg = original_load_config(*args, **kwargs)
        cfg["logging"]["predictions_log_path"] = str(log_path)
        return cfg

    monkeypatch.setattr("src.utils.drift_check.load_config", _patched_load_config)

    results = run_drift_check()
    assert len(results) > 0
    for r in results:
        assert "feature" in r and "p_value" in r and "drifted" in r
