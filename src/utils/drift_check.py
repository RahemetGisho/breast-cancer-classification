"""Lightweight data drift check.

Models degrade silently when the distribution of incoming data shifts
away from what they were trained on — nobody gets an error, the model
just quietly gets worse. This script is the minimum viable defense: for
each feature, run a Kolmogorov-Smirnov two-sample test comparing the
logged production request distribution against the training
distribution, and flag features where they've diverged significantly.

This is not a full observability stack. It's a hook proving the
awareness that one is needed — run it periodically (e.g. weekly) against
the growing predictions.jsonl log.

Usage: python -m src.utils.drift_check
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from src.data.load_data import load_validated_dataframe
from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Below this p-value, we consider the feature's distribution to have
# meaningfully shifted. 0.01 rather than the conventional 0.05 to avoid
# flagging on small production sample sizes where noise alone can trip
# a looser threshold.
DRIFT_P_VALUE_THRESHOLD = 0.01


def load_production_requests(log_path: Path) -> pd.DataFrame:
    """Parse the JSONL prediction log into a DataFrame of the request
    features that were actually sent to the API in production.
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log found at {log_path}. The API needs to have "
            "served at least a few requests before a drift check is meaningful."
        )

    rows = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            rows.append(entry["request"])

    return pd.DataFrame(rows)


def check_drift(production_df: pd.DataFrame, reference_df: pd.DataFrame) -> list[dict]:
    """Run a KS test per feature, comparing production vs reference
    (training) distributions. Returns a list of per-feature results,
    each flagged as drifted or not.
    """
    results = []
    for feature in FEATURE_COLUMNS:
        if feature not in production_df.columns:
            continue

        statistic, p_value = ks_2samp(reference_df[feature], production_df[feature])
        drifted = p_value < DRIFT_P_VALUE_THRESHOLD

        results.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": bool(drifted),
            }
        )

    return results


def run_drift_check() -> list[dict]:
    config = load_config()
    log_path = resolve_path(config["logging"]["predictions_log_path"])

    reference_df = load_validated_dataframe().drop(columns=[TARGET_COLUMN])
    production_df = load_production_requests(log_path)

    if len(production_df) < 30:
        logger.warning(
            "Only %d production requests logged — drift results with this few "
            "samples are unreliable. Treat as directional, not conclusive.",
            len(production_df),
        )

    results = check_drift(production_df, reference_df)

    drifted_features = [r["feature"] for r in results if r["drifted"]]
    if drifted_features:
        logger.warning(
            "Drift detected in %d feature(s): %s",
            len(drifted_features),
            drifted_features,
        )
    else:
        logger.info("No significant drift detected across %d features.", len(results))

    return results


if __name__ == "__main__":
    for result in run_drift_check():
        flag = "DRIFTED" if result["drifted"] else "ok"
        print(f"{result['feature']:30s} p={result['p_value']:.4f}  [{flag}]")
