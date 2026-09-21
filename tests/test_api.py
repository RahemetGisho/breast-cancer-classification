"""Tests for the FastAPI serving layer.

Uses FastAPI's TestClient (no real network/socket needed) against the
already-trained model in the registry. Run training first if this is a
fresh checkout — that's exactly the dependency ordering CI enforces.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_payload():
    # A real row from the dataset (first malignant case) — using real,
    # physically plausible values rather than arbitrary numbers.
    from src.data.load_data import load_validated_dataframe
    from src.data.schema import TARGET_COLUMN

    df = load_validated_dataframe()
    row = df.drop(columns=[TARGET_COLUMN]).iloc[0]
    return row.to_dict()


def test_health_endpoint_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response_shape(sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("benign", "malignant")
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert len(body["top_contributors"]) == 5
    for contributor in body["top_contributors"]:
        assert contributor["direction"] in (
            "increases_malignant_risk",
            "decreases_malignant_risk",
        )


def test_predict_rejects_missing_field(sample_payload):
    incomplete = dict(sample_payload)
    del incomplete["radius_mean"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_predict_rejects_negative_value(sample_payload):
    broken = dict(sample_payload)
    broken["area_mean"] = -50.0
    response = client.post("/predict", json=broken)
    assert response.status_code == 422


def test_predict_logs_each_request(sample_payload, tmp_path, monkeypatch):
    """The prediction log is what the drift-check script reads later —
    prove an entry is actually written per request."""
    from src.utils import config as config_module

    log_path = tmp_path / "predictions.jsonl"
    original_load_config = config_module.load_config

    def _patched_load_config(*args, **kwargs):
        cfg = original_load_config(*args, **kwargs)
        # Path./ with an absolute right-hand side returns that absolute
        # path unchanged, so this safely overrides regardless of
        # resolve_path()'s PROJECT_ROOT join.
        cfg["logging"]["predictions_log_path"] = str(log_path)
        return cfg

    monkeypatch.setattr("api.main.load_config", _patched_load_config)

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    assert log_path.exists()
    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) == 1


def test_health_reports_model_not_found_when_bundle_missing(monkeypatch):
    """Simulates a fresh deployment before training has run: get_model_bundle
    raises FileNotFoundError, and /health should report that clearly
    (status 200, model_loaded=False) rather than crashing.
    """

    def _raise_not_found():
        raise FileNotFoundError("no model registered yet")

    monkeypatch.setattr("api.main.get_model_bundle", _raise_not_found)

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "model_not_found"
    assert body["model_loaded"] is False


def test_predict_returns_503_when_model_not_loaded(sample_payload, monkeypatch):
    def _raise_not_found():
        raise FileNotFoundError("no model registered yet")

    monkeypatch.setattr("api.main.get_model_bundle", _raise_not_found)

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 503


def test_predict_returns_500_on_unexpected_prediction_error(
    sample_payload, monkeypatch
):
    """A model that loads fine but fails during inference (corrupt
    artifact, incompatible sklearn version, etc.) should surface as a
    clean 500, not leak a raw traceback to the client.
    """

    class _BrokenBundle:
        def predict_one(self, features):
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr("api.main.get_model_bundle", lambda: _BrokenBundle())

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
