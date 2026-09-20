"""Centralized logging configuration.

Import get_logger(__name__) in any module instead of calling
logging.basicConfig() ad hoc in multiple places, which causes duplicate
or inconsistent log output once more than one module configures logging.
"""

import logging
import sys

_CONFIGURED = False


def _configure_root(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        # Fall back to configs/config.yaml's logging.level if the caller
        # didn't specify one explicitly, so changing that one value
        # actually takes effect instead of sitting unused in the config.
        try:
            from src.utils.config import load_config

            level = load_config()["logging"]["level"]
        except Exception:
            level = "INFO"

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    _configure_root(level)
    return logging.getLogger(name)
