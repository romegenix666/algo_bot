"""Structured logging configured once at import time.

Use ``from src.utils.logging import logger`` everywhere. Avoids the global
``logging`` module's surprises and gives us machine-readable JSON logs that we
can ship to a file + stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from src.utils.settings import PROJECT_ROOT, settings

_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure() -> None:
    _logger.remove()

    _logger.add(
        sys.stdout,
        level=settings.env.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,
    )

    log_path: Path = _LOG_DIR / "algobot.log"
    _logger.add(
        log_path,
        level=settings.env.log_level,
        rotation=settings.get("logging", "rotation", default="10 MB"),
        retention=settings.get("logging", "retention", default="30 days"),
        serialize=True,  # JSON lines
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


_configure()
logger = _logger

__all__ = ["logger"]
