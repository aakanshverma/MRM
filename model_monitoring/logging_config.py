"""Logging setup with an in-memory collector for the Excel exception log.

Two things happen here:

1.  A conventional console + rotating-file logger is configured.
2.  Every WARNING/ERROR (and any explicitly recorded exception) is captured in
    a :class:`ExecutionLog`, which the Excel writer renders on the
    "Execution Log" sheet.  Governance requires that warnings surfaced during a
    run are visible to the model owner, not buried in a terminal.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER_NAME = "model_monitoring"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


# ---------------------------------------------------------------------------
# Execution log
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    """One recorded event destined for the governance sheet."""

    timestamp: str
    level: str
    stage: str
    message: str
    detail: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "TIMESTAMP": self.timestamp,
            "LEVEL": self.level,
            "STAGE": self.stage,
            "MESSAGE": self.message,
            "DETAIL": self.detail,
        }


@dataclass
class ExecutionLog:
    """Collects warnings, errors and stage timings for the Excel report."""

    entries: List[LogEntry] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    _current_stage: str = "INIT"

    # -- stage tracking ----------------------------------------------------
    def set_stage(self, stage: str) -> None:
        self._current_stage = stage

    @property
    def stage(self) -> str:
        return self._current_stage

    # -- recording ---------------------------------------------------------
    def record(self, level: str, message: str, detail: str = "", stage: Optional[str] = None) -> None:
        self.entries.append(
            LogEntry(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                level=str(level).upper(),
                stage=stage or self._current_stage,
                message=str(message),
                detail=str(detail),
            )
        )

    def info(self, message: str, detail: str = "") -> None:
        self.record("INFO", message, detail)

    def warning(self, message: str, detail: str = "") -> None:
        self.record("WARNING", message, detail)

    def error(self, message: str, detail: str = "") -> None:
        self.record("ERROR", message, detail)

    def exception(self, message: str, exc: BaseException) -> None:
        self.record(
            "ERROR",
            message,
            "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )

    # -- queries -----------------------------------------------------------
    def count(self, level: str) -> int:
        target = level.upper()
        return sum(1 for entry in self.entries if entry.level == target)

    @property
    def warning_count(self) -> int:
        return self.count("WARNING")

    @property
    def error_count(self) -> int:
        return self.count("ERROR")

    def to_rows(self) -> List[Dict[str, Any]]:
        return [entry.as_row() for entry in self.entries]


class _CollectingHandler(logging.Handler):
    """Mirrors WARNING+ log records into an :class:`ExecutionLog`."""

    def __init__(self, execution_log: ExecutionLog, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self.execution_log = execution_log

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin
        try:
            detail = ""
            if record.exc_info:
                detail = "".join(traceback.format_exception_only(*record.exc_info[:2])).strip()
            self.execution_log.record(record.levelname, record.getMessage(), detail)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    *,
    execution_log: Optional[ExecutionLog] = None,
    quiet: bool = False,
) -> tuple[logging.Logger, ExecutionLog]:
    """Configure the package logger.

    Parameters
    ----------
    level
        Console log level name (``DEBUG``/``INFO``/``WARNING``/...).
    log_file
        Optional path for a rotating file handler (always DEBUG level).
    execution_log
        Reuse an existing collector; a new one is created when omitted.
    quiet
        Suppress the console handler (used by the test suite).

    Returns
    -------
    (logger, execution_log)
    """
    collector = execution_log if execution_log is not None else ExecutionLog()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):  # idempotent re-configuration
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover
            pass

    if not quiet:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    logger.addHandler(_CollectingHandler(collector))
    return logger, collector


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the package logger."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


__all__ = ["setup_logging", "get_logger", "ExecutionLog", "LogEntry", "LOGGER_NAME"]
