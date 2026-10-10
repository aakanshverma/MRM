"""Shared fixtures.

Fixtures are deliberately small and hand-constructed: a test that fails should
point at one behaviour, not require unpicking a 25,000-row generator.  The
full-scale generator is exercised separately in ``test_pipeline.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import SEGMENTS, make_config, make_frame  # noqa: E402

from model_monitoring.config import MonitoringConfig  # noqa: E402
from model_monitoring.data_validation import prepare_frame  # noqa: E402
from model_monitoring.logging_config import setup_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_logging():
    """Keep the package logger silent during tests but still collecting."""
    logger, execution_log = setup_logging("ERROR", quiet=True)
    yield execution_log


@pytest.fixture
def reference_counts() -> dict:
    """A simple reference distribution used by the hand-calculation tests."""
    return {"CAT A": 100, "CAT B": 200, "CAT C1": 300, "CAT C2": 200, "CAT D": 150, "CAT E": 50}


@pytest.fixture
def shifted_counts() -> dict:
    return {"CAT A": 50, "CAT B": 150, "CAT C1": 300, "CAT C2": 250, "CAT D": 200, "CAT E": 50}


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def config() -> MonitoringConfig:
    return make_config()


@pytest.fixture
def prepared(frame, config) -> pd.DataFrame:
    config.resolve_variables(list(frame.columns))
    return prepare_frame(frame, config)


@pytest.fixture
def pipeline_result(frame, config):
    """A full in-memory pipeline run (no Excel) for integration assertions."""
    from model_monitoring.main import run_pipeline

    return run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *args, **kwargs: None
    )
