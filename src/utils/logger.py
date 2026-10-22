"""Centralized logging for the advisor-risk-mlops project.

Usage:
    from src.utils import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with consistent formatting.

    Args:
        name:  Logger name — typically ``__name__`` from the calling module.
        level: Logging level (default ``INFO``).

    Returns:
        A ``logging.Logger`` instance with a stream handler attached.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
