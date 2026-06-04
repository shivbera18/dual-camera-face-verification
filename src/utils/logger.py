"""Colored console + rotating file logger."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import colorlog

from src.utils.config import REPO_ROOT


_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_console_handler(level: int) -> logging.Handler:
    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s" + _DEFAULT_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    return handler


def _build_file_handler(log_file: Path, level: int) -> logging.Handler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(_build_console_handler(level))
    if log_file is not None:
        lf = Path(log_file)
        if not lf.is_absolute():
            lf = REPO_ROOT / lf
        logger.addHandler(_build_file_handler(lf, level))
    return logger
