"""Export local CSV snapshots of the dataset, for manual inspection only.

Nothing in the actual pipeline (train.py, tune.py, the API) reads these
files -- they still load fresh from load_breast_cancer() every time,
per src/data/load_data.py. This exists purely so you can open the data
in a spreadsheet or diff it by eye without going through Python.
"""

import pandas as pd

from src.data.load_data import load_raw_dataframe, load_validated_dataframe
from src.utils.config import resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def export_raw_snapshot() -> None:
    """Raw, unvalidated data -- exactly what load_breast_cancer() gives us,
    with just our column-naming convention applied. 'Raw' here means
    'raw relative to our validation step', not raw sensor data."""
    df = load_raw_dataframe()
    path = resolve_path("data/raw/breast_cancer_raw.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote raw snapshot: %d rows to %s", len(df), path)


def export_processed_snapshot() -> None:
    """The validated version -- passed through the pandera schema, exactly
    what train.py actually trains on."""
    df = load_validated_dataframe()
    path = resolve_path("data/processed/breast_cancer_validated.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote processed snapshot: %d rows to %s", len(df), path)


if __name__ == "__main__":
    export_raw_snapshot()
    export_processed_snapshot()
