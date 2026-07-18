"""Minimal project-level logging configuration.

Usage:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Loading PDFs from %s", pdf_dir)
    logger.warning("Skipping corrupted page %d", page_num)
    logger.error("Embedding failed: %s", err)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with sane defaults for CLI / batch scripts.

    - Log level defaults to INFO (override with env LOG_LEVEL).
    - Writes to stderr with timestamps.
    - Does NOT capture or redirect existing `print()` calls.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    level = logging.getLevelNamesMapping().get(
        (__import__("os").getenv("LOG_LEVEL") or "INFO").upper(),
        logging.INFO,
    )
    logger.setLevel(level)
    logger.propagate = False

    return logger
