import pandas as pd
import pytest

from src.data.load_data import load_raw_dataframe, load_validated_dataframe
from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.validate import DataValidationError, validate_dataframe


@pytest.fixture(scope="module")
def valid_df() -> pd.DataFrame:
    return load_raw_dataframe()


def test_valid_data_passes(valid_df):
    """Sanity check: real data should pass without raising."""
    validated = validate_dataframe(valid_df)
    assert len(validated) == len(valid_df)


def test_load_validated_dataframe_returns_expected_shape():
    df = load_validated_dataframe()
    assert df.shape == (569, len(FEATURE_COLUMNS) + 1)
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


def test_missing_column_is_rejected(valid_df):
    broken = valid_df.drop(columns=["radius_mean"])
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_unexpected_extra_column_is_rejected(valid_df):
    broken = valid_df.copy()
    broken["unexpected_column"] = 1.0
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_null_values_are_rejected(valid_df):
    broken = valid_df.copy()
    broken.loc[0, "radius_mean"] = None
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_negative_feature_value_is_rejected(valid_df):
    """Physical measurements (radius, area, etc.) cannot be negative —
    a negative value signals a corrupt upstream feed, not valid data."""
    broken = valid_df.copy()
    broken.loc[0, "area_mean"] = -100.0
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_invalid_target_value_is_rejected(valid_df):
    broken = valid_df.copy()
    broken.loc[0, TARGET_COLUMN] = 2  # only 0/1 are valid
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_out_of_range_value_is_rejected(valid_df):
    """A value far outside plausible physical range (e.g. a unit error:
    reporting mm instead of the expected scale) should be caught."""
    broken = valid_df.copy()
    broken.loc[0, "radius_mean"] = 999_999.0
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)
