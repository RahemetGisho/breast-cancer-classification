import pandas as pd
import pandera as pa

from src.data.schema import RAW_SCHEMA
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when a DataFrame does not conform to the expected schema."""


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error("Data validation failed:\n%s", exc.failure_cases)
        raise DataValidationError(
            f"Data failed schema validation with {len(exc.failure_cases)} failure case(s). "
            f"See failure_cases for details: {exc.failure_cases.to_dict('records')}"
        ) from exc

    logger.info("Data validation passed: %d rows, %d columns", *validated.shape)
    return validated
