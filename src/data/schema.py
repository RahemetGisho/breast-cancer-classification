import os

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import pandera as pa  # noqa: E402, F401
from pandera import Check, Column, DataFrameSchema  # noqa: E402

# The 30 WDBC features: 10 base measurements x 3 aggregations (mean, se, worst).
_BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
_SUFFIXES = ["mean", "se", "worst"]

FEATURE_COLUMNS = [
    f"{base}_{suffix}" for suffix in _SUFFIXES for base in _BASE_FEATURES
]

TARGET_COLUMN = "target"


_FEATURE_CHECK = Check.in_range(min_value=0.0, max_value=10_000.0)

RAW_SCHEMA = DataFrameSchema(
    columns={
        **{
            col: Column(float, checks=_FEATURE_CHECK, nullable=False)
            for col in FEATURE_COLUMNS
        },
        TARGET_COLUMN: Column(int, checks=Check.isin([0, 1]), nullable=False),
    },
    strict=True,  # reject unexpected extra columns rather than silently ignoring them
    coerce=True,
)
