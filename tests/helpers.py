"""Dataset and configuration builders shared by the test modules.

Kept out of ``conftest.py`` so individual tests can import them directly when
they need a variant of the standard fixtures (a shorter span, no outcome
column, a different reference window).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_monitoring.config import (  # noqa: E402
    DataSourceConfig,
    ModelConfig,
    MonitoringConfig,
    OutcomeConfig,
    ReferencePeriodConfig,
    SegmentConfig,
    VariableSpec,
)

SEGMENTS = ["CAT A", "CAT B", "CAT C1", "CAT C2", "CAT D", "CAT E"]


def make_frame(
    n: int = 2_400,
    *,
    seed: int = 7,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    drift: bool = True,
    with_outcome: bool = True,
) -> pd.DataFrame:
    """Two-scorecard extract spanning eight complete quarters."""
    rng = np.random.default_rng(seed)
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    offsets = np.linspace(0, span, n).astype(int)
    login = pd.Timestamp(start) + pd.to_timedelta(offsets, unit="D")
    clock = offsets / max(1, span)

    # MODEL_1 drifts toward risk, MODEL_2 is stable
    drift_strength = 2.0 if drift else 0.0
    latent_1 = rng.normal(0, 1, n) + drift_strength * clock
    latent_2 = rng.normal(0, 1, n)
    cuts = np.quantile(latent_1[: n // 4], [0.22, 0.46, 0.66, 0.81, 0.93])
    segment_1 = np.array(SEGMENTS)[np.clip(np.searchsorted(cuts, latent_1), 0, 5)]
    cuts_2 = np.quantile(latent_2[: n // 4], [0.22, 0.46, 0.66, 0.81, 0.93])
    segment_2 = np.array(SEGMENTS)[np.clip(np.searchsorted(cuts_2, latent_2), 0, 5)]

    frame = pd.DataFrame({
        "LAN_ID": [f"L{i:06d}" for i in range(n)],
        # repeat customers so migration has something to match on
        "CUSTOMER_ID": [f"C{i % (n // 2):06d}" for i in range(n)],
        "LOGIN_DATE": login,
        "MODEL_1_SEGMENT": segment_1,
        "MODEL_2_SEGMENT": segment_2,
        "MODEL_1_VARIABLE_1": rng.normal(600, 80, n) - 120 * clock,
        "MODEL_1_VARIABLE_2": rng.choice(["SALARIED", "SELF_EMP", "PROF"], n),
        "MODEL_1_VARIABLE_1_SUBSCORE": rng.integers(0, 50, n).astype("float64"),
        "MODEL_2_VARIABLE_1": rng.gamma(3.0, 12.0, n),
        "MODEL_1_TOTAL_SCORE": np.round(700 - 40 * latent_1),
    })
    if with_outcome:
        probability = 1 / (1 + np.exp(-(-2.5 + 0.7 * latent_1)))
        frame["DEFAULT_FLAG"] = (rng.random(n) < probability).astype("int8")
        frame["EXPECTED_PD"] = np.round(probability, 5)
    return frame


def make_config(
    *,
    with_outcome: bool = True,
    reference_1: tuple = ("2024-01-01", "2024-06-30"),
    reference_2: tuple = ("2024-07-01", "2024-12-31"),
    min_reference_records: int = 100,
    min_monitoring_records: int = 50,
) -> MonitoringConfig:
    """Two models with deliberately DIFFERENT reference windows."""
    config = MonitoringConfig(
        models=[
            ModelConfig(
                name="MODEL_1",
                segment_column="MODEL_1_SEGMENT",
                score_column="MODEL_1_TOTAL_SCORE",
                variables=[
                    VariableSpec(
                        name="MODEL_1_VARIABLE_1", var_type="NUMERIC", n_bins=5,
                        subscore_column="MODEL_1_VARIABLE_1_SUBSCORE",
                    ),
                    VariableSpec(name="MODEL_1_VARIABLE_2", var_type="CATEGORICAL"),
                ],
                reference=ReferencePeriodConfig(
                    model="MODEL_1", start_date=reference_1[0], end_date=reference_1[1],
                    mode="DEVELOPMENT",
                ),
            ),
            ModelConfig(
                name="MODEL_2",
                segment_column="MODEL_2_SEGMENT",
                variables=[VariableSpec(name="MODEL_2_VARIABLE_1", var_type="NUMERIC", n_bins=5)],
                reference=ReferencePeriodConfig(
                    model="MODEL_2", start_date=reference_2[0], end_date=reference_2[1],
                    mode="PRODUCTION_BASELINE",
                ),
            ),
        ],
        data_source=DataSourceConfig(),
    )
    config.thresholds.min_reference_records = min_reference_records
    config.thresholds.min_monitoring_records = min_monitoring_records
    config.thresholds.min_variable_records = 20
    config.thresholds.min_outcome_records = 50
    config.thresholds.min_bad_count = 5
    config.runtime.non_interactive = True
    config.runtime.min_quarter_records = 10
    if with_outcome:
        config.outcome = OutcomeConfig(
            outcome_column="DEFAULT_FLAG", pd_column="EXPECTED_PD",
            observation_window_months=12,
        )
    return config
