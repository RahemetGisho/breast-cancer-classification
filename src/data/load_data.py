import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.validate import validate_dataframe
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _sklearn_name_to_schema_name(name: str) -> str:

    parts = name.split(" ")
    if parts[0] == "mean":
        suffix, base_parts = "mean", parts[1:]
    elif parts[0] == "worst":
        suffix, base_parts = "worst", parts[1:]
    elif parts[-1] == "error":
        suffix, base_parts = "se", parts[:-1]
    else:
        raise ValueError(f"Unrecognized sklearn feature name format: {name!r}")

    base = "_".join(base_parts)
    return f"{base}_{suffix}"


def load_raw_dataframe() -> pd.DataFrame:

    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df.columns = [_sklearn_name_to_schema_name(c) for c in df.columns]
    df = df[FEATURE_COLUMNS]

    # sklearn: 0=malignant, 1=benign -> flip so 1=malignant (our positive class)
    df[TARGET_COLUMN] = 1 - bunch.target

    logger.info(
        "Loaded raw dataset: %d rows, malignant=%d, benign=%d",
        len(df),
        int(df[TARGET_COLUMN].sum()),
        int((df[TARGET_COLUMN] == 0).sum()),
    )
    return df


def load_validated_dataframe() -> pd.DataFrame:

    raw = load_raw_dataframe()
    return validate_dataframe(raw)


if __name__ == "__main__":
    df = load_validated_dataframe()
    print(df.head())
    print(df[TARGET_COLUMN].value_counts())
