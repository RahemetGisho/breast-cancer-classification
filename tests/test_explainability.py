"""Tests for the explainability module.

Focus: the local top-contributors function returns a sane, correctly
ranked, correctly signed explanation — not just "doesn't crash."
"""

import numpy as np

from src.explainability.shap_explain import top_contributors_for_instance


def test_top_contributors_ranks_by_absolute_value():
    shap_row = np.array([0.5, -0.9, 0.1, 0.3, -0.2])
    features = ["a", "b", "c", "d", "e"]

    result = top_contributors_for_instance(shap_row, features, top_n=3)

    assert [r["feature"] for r in result] == ["b", "a", "d"]


def test_top_contributors_direction_labels_are_correct():
    shap_row = np.array([0.8, -0.6])
    features = ["raises_risk", "lowers_risk"]

    result = top_contributors_for_instance(shap_row, features, top_n=2)
    by_feature = {r["feature"]: r for r in result}

    assert by_feature["raises_risk"]["direction"] == "increases_malignant_risk"
    assert by_feature["lowers_risk"]["direction"] == "decreases_malignant_risk"


def test_top_contributors_respects_top_n():
    shap_row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    features = [f"f{i}" for i in range(5)]

    result = top_contributors_for_instance(shap_row, features, top_n=2)

    assert len(result) == 2
