"""Load the project's central YAML config.

Every module reads settings through this function instead of hardcoding
paths, thresholds, or hyperparameters. This keeps behavior configurable
without touching code, and makes it trivial to point at a different
config file in tests.
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the project config as a dict.

    Args:
        config_path: Optional override path. Defaults to configs/config.yaml
            at the project root.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the file is empty or not valid YAML mapping.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse into a mapping")

    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from the config relative to the project root."""
    return PROJECT_ROOT / relative_path
