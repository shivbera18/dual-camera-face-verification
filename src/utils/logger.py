from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import colorlog
except ImportError:  # pragma: no cover - fallback for minimal environments
    colorlog = None

from src.utils.config import resolve_project_path

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, log_file: str | Path | None = None, level: str | int = "INFO") -> logging.Logger:
    """Return a configured logger with console and optional rotating file output."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if colorlog is not None:
        console_formatter: logging.Formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = resolve_project_path(log_file)
    else:
        path = resolve_project_path("artifacts/logs") / f"{name.replace('.', '_')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
