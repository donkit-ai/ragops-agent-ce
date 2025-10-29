from __future__ import annotations

import sys

from loguru import logger

from .config import Settings, load_settings

_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


def setup_logging(settings: Settings | None = None) -> None:
    """Configure loguru logger according to .env / Settings.

    Priority of level:
    1) settings.log_level if provided
    2) env RAGOPS_LOG_LEVEL
    3) default INFO
    """
    s = settings or load_settings()
    level = s.log_level

    # Reset existing handlers and set our sink
    try:
        logger.remove()
    except Exception:
        pass

    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=_LOGURU_FORMAT,
    )
