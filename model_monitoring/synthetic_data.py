"""Synthetic NBFC scorecard extract for testing and demonstration.

The generator is not decoration: it is the test oracle.  It deliberately plants
every condition the pipeline claims to handle, so that a run against it
exercises the real code paths rather than a happy path.

Planted scenarios (see :data:`SCENARIOS`)
-----------------------------------------
1.  Stable segment distribution (low PSI)                      -- MODEL_2
2.  Substantial distribution shift (high PSI)                  -- MODEL_1
3.  A segment with zero observations in later quarters          -- MODEL_3
4.  Missing and invalid segment values                          -- MODEL_1, MODEL_4
5.  Missing numeric and categorical features                    -- several
6.  An unseen category appearing mid-series                     -- MODEL_3
7.  Different reference periods per scorecard                    -- configuration
8.  An incomplete final quarter                                  -- all models
9.  A characteristic with substantial CSI                        -- MODEL_1_VARIABLE_1/3
10. Customer upgrades and downgrades across quarters             -- repeat applicants
11. Duplicate LAN / customer records                             -- planted duplicates
12. Insufficient reference and monitoring observations           -- MODEL_4

Segments are *derived* from the characteristics through fixed cut-offs rather
than sampled directly, so PSI, CSI and migration tell a mutually consistent
story the way they do in a real portfolio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SEGMENT_ORDER

SCENARIOS: List[Dict[str, str]] = [
    {"ID": "1", "SCENARIO": "Stable segment distribution (low PSI)",
     "WHERE": "MODEL_2", "EXPECT": "PSI below 0.10 in every quarter; severity GREEN."},
    {"ID": "2", "SCENARIO": "Substantial segment distribution shift (high PSI)",
     "WHERE": "MODEL_1", "EXPECT": "PSI rises past 0.10 then 0.25; severity AMBER then RED."},
    {"ID": "3", "SCENARIO": "Segment with zero observations",
     "WHERE": "MODEL_3 stops emitting the riskiest grade from 2025Q3",
     "EXPECT": "All six segments still appear in the PSI table; the empty one is handled by the "
               "documented smoothing rule rather than a division error."},
    {"ID": "4", "SCENARIO": "Missing and invalid segment values",
     "WHERE": "MODEL_1 (~1.5% missing, ~0.4% invalid), MODEL_4 (higher)",
     "EXPECT": "MER missing/invalid counts populated; DQ-041/DQ-042 raised; those records "
               "excluded from PSI."},
    {"ID": "5", "SCENARIO": "Missing numeric and categorical features",
     "WHERE": "MODEL_1_VARIABLE_2 / MODEL_2_VARIABLE_3 / MODEL_3_VARIABLE_2",
     "EXPECT": "MISSING bin populated in CSI bin detail; missingness reported per quarter."},
    {"ID": "6", "SCENARIO": "Unseen category",
     "WHERE": "MODEL_3_VARIABLE_2 gains 'GIG_ECONOMY' from 2025Q3",
     "EXPECT": "UNSEEN bin populated, unseen-category alert raised."},
    {"ID": "7", "SCENARIO": "Different reference period per scorecard",
     "WHERE": "config/config.sample.yaml",
     "EXPECT": "Each model's PSI/CSI uses its own window; the Configuration sheet shows four "
               "different windows and two different reference modes."},
    {"ID": "8", "SCENARIO": "Incomplete quarter",
     "WHERE": "the extract ends mid-quarter",
     "EXPECT": "Final quarter flagged INCOMPLETE and excluded from the primary comparison."},
    {"ID": "9", "SCENARIO": "Characteristic with substantial CSI",
     "WHERE": "MODEL_1_VARIABLE_1 drifts; MODEL_1_VARIABLE_3 missingness jumps",
     "EXPECT": "CSI above 0.25 for those characteristics in later quarters."},
    {"ID": "10", "SCENARIO": "Customer upgrades and downgrades",
     "WHERE": "~18% of customers re-apply in a later quarter",
     "EXPECT": "Migration matrices populated with non-zero off-diagonal flows."},
    {"ID": "11", "SCENARIO": "Duplicate LAN / customer records",
     "WHERE": "~0.8% of LANs duplicated; some customer+date duplicates",
     "EXPECT": "DQ-001/DQ-002 duplicate diagnostics raised."},
    {"ID": "12", "SCENARIO": "Insufficient reference / monitoring observations",
     "WHERE": "MODEL_4 goes live 2025-07 with low volume",
     "EXPECT": "Reference adequacy alert; MODEL_4 stability metrics labelled GREY, not GREEN."},
]

# Target reference-period segment shares (safest -> riskiest).
_TARGET_SHARES = np.array([0.22, 0.24, 0.20, 0.15, 0.12, 0.07])


@dataclass
class GeneratorConfig:
    """Knobs for the synthetic extract."""

    n_rows: int = 25_000
    start_date: date = date(2024, 1, 1)
    #: Deliberately mid-quarter so the final quarter is incomplete.
    end_date: date = date(2026, 9, 20)
    seed: int = 42
    repeat_customer_share: float = 0.18
    duplicate_lan_share: float = 0.008
    outcome_window_months: int = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cutoffs(values: np.ndarray, shares: np.ndarray = _TARGET_SHARES) -> np.ndarray:
    """Fixed score cut-offs reproducing ``shares`` on the reference sample."""
    quantiles = np.cumsum(shares)[:-1]
    return np.quantile(values[np.isfinite(values)], quantiles)


def _assign_segments(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Map a latent risk score to the ordered segment list (higher = riskier)."""
    indices = np.searchsorted(cutoffs, values, side="right")
    return np.array(SEGMENT_ORDER, dtype=object)[np.clip(indices, 0, len(SEGMENT_ORDER) - 1)]


def _inject_missing(
    rng: np.random.Generator, values: np.ndarray, rate: float,
    *, mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = values.astype("float64", copy=True) if values.dtype.kind in "iuf" else values.astype(object)
    draw = rng.random(len(values)) < rate
    if mask is not None:
        draw &= mask
    if out.dtype.kind == "f":
        out[draw] = np.nan
    else:
        out[draw] = None
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    config: Optional[GeneratorConfig] = None, **overrides: Any
) -> pd.DataFrame:
    """Build the synthetic customer-level scorecard extract."""
    settings = config or GeneratorConfig()
    for key, value in overrides.items():
        setattr(settings, key, value)
    rng = np.random.default_rng(settings.seed)
    n = int(settings.n_rows)

    # -- dates (mild seasonal growth so volumes are not flat) --------------
    span_days = (settings.end_date - settings.start_date).days
    weights = 1.0 + 0.35 * np.sin(np.linspace(0, 6 * np.pi, span_days + 1)) \
        + np.linspace(0, 0.5, span_days + 1)
    weights = np.clip(weights, 0.05, None)
    weights /= weights.sum()
    offsets = rng.choice(span_days + 1, size=n, p=weights)
    login = pd.to_datetime(settings.start_date) + pd.to_timedelta(offsets, unit="D")
    login = pd.Series(login).sort_values().reset_index(drop=True)
    #: 0 at the start of the window, 1 at the end -- the drift clock.
    t = ((login - login.min()).dt.days / max(1, span_days)).to_numpy()
    quarter = login.dt.to_period("Q").astype("string").to_numpy()

    frame = pd.DataFrame({
        "LAN_ID": [f"LAN{idx:08d}" for idx in range(1, n + 1)],
        "LOGIN_DATE": login,
    })

    # -- customers, with repeat applicants for migration -------------------
    n_customers = max(1, int(n * (1.0 - settings.repeat_customer_share)))
    customer_index = rng.integers(0, n_customers, size=n)
    frame["CUSTOMER_ID"] = [f"CUST{idx:07d}" for idx in customer_index]

    # -- portfolio attributes ----------------------------------------------
    frame["PRODUCT"] = rng.choice(
        ["PERSONAL_LOAN", "TWO_WHEELER", "LAP", "BUSINESS_LOAN"], size=n,
        p=[0.42, 0.18, 0.22, 0.18],
    )
    frame["CHANNEL"] = rng.choice(
        ["BRANCH", "DSA", "DIGITAL", "PARTNER"], size=n, p=[0.30, 0.28, 0.30, 0.12]
    )
    frame["BRANCH_STATE"] = rng.choice(
        ["MH", "KA", "TN", "GJ", "UP", "DL", "RJ", "WB"], size=n
    )
    frame["SANCTION_AMOUNT"] = np.round(
        rng.lognormal(mean=12.0, sigma=0.65, size=n) / 1000.0
    ) * 1000.0

    latent: Dict[str, np.ndarray] = {}

    # ==================================================================
    # MODEL_1 -- unsecured personal loan scorecard.  DRIFTS (scenarios 2, 9)
    # ==================================================================
    bureau = np.clip(rng.normal(720 - 55 * t, 70), 300, 900)            # falls over time
    foir = np.clip(rng.normal(38 + 14 * t, 11), 2, 95)                  # rises over time
    vintage_months = np.clip(rng.gamma(4.0, 14.0, n) - 18 * t, 0, 300)
    enquiries = rng.poisson(1.6 + 1.9 * t, n)
    employment = np.where(
        rng.random(n) < (0.62 - 0.22 * t), "SALARIED",
        np.where(rng.random(n) < 0.72, "SELF_EMPLOYED", "SELF_EMPLOYED_PROF"),
    )
    residence = rng.choice(["OWNED", "RENTED", "PARENTAL", "COMPANY"], size=n,
                           p=[0.38, 0.40, 0.18, 0.04])

    m1_latent = (
        -0.016 * (bureau - 700)
        + 0.055 * (foir - 38)
        - 0.012 * (vintage_months - 50)
        + 0.30 * enquiries
        + np.where(employment == "SALARIED", -0.45, 0.35)
        + np.where(residence == "OWNED", -0.30, 0.10)
        + rng.normal(0, 0.9, n)
    )
    latent["MODEL_1"] = m1_latent
    frame["MODEL_1_VARIABLE_1"] = np.round(bureau).astype("float64")
    frame["MODEL_1_VARIABLE_2"] = np.round(foir, 2)
    frame["MODEL_1_VARIABLE_3"] = np.round(vintage_months, 1)
    frame["MODEL_1_VARIABLE_4"] = enquiries.astype("float64")
    frame["MODEL_1_VARIABLE_5"] = employment
    frame["MODEL_1_VARIABLE_6"] = residence

    # scenario 5: ordinary missingness in a numeric and a categorical input
    frame["MODEL_1_VARIABLE_2"] = _inject_missing(rng, frame["MODEL_1_VARIABLE_2"].to_numpy(), 0.04)
    frame["MODEL_1_VARIABLE_6"] = _inject_missing(rng, frame["MODEL_1_VARIABLE_6"].to_numpy(object), 0.03)
    # scenario 9b: a feature-pipeline break -- missingness jumps from 3% to 28%
    break_mask = login.to_numpy() >= np.datetime64("2026-01-01")
    base_missing = rng.random(n) < 0.03
    broken_missing = (rng.random(n) < 0.28) & break_mask
    variable_3 = frame["MODEL_1_VARIABLE_3"].to_numpy(dtype="float64", copy=True)
    variable_3[base_missing | broken_missing] = np.nan
    frame["MODEL_1_VARIABLE_3"] = variable_3

    # ==================================================================
    # MODEL_2 -- loan against property scorecard.  STABLE (scenario 1)
    # ==================================================================
    ltv = np.clip(rng.normal(62, 12, n), 10, 95)
    property_age = np.clip(rng.gamma(3.0, 4.0, n), 0, 60)
    income = rng.lognormal(11.4, 0.55, n)
    obligations = np.clip(rng.normal(0.28, 0.12, n), 0, 1.2)
    property_type = rng.choice(["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL"], size=n,
                               p=[0.72, 0.24, 0.04])
    m2_latent = (
        0.045 * (ltv - 62)
        + 0.020 * (property_age - 12)
        - 0.60 * (np.log(income) - 11.4)
        + 2.1 * (obligations - 0.28)
        + np.where(property_type == "RESIDENTIAL", -0.20, 0.25)
        + rng.normal(0, 1.0, n)
    )
    latent["MODEL_2"] = m2_latent
    frame["MODEL_2_VARIABLE_1"] = np.round(ltv, 2)
    frame["MODEL_2_VARIABLE_2"] = np.round(property_age, 1)
    frame["MODEL_2_VARIABLE_3"] = np.round(income, 0)
    frame["MODEL_2_VARIABLE_4"] = np.round(obligations, 4)
    frame["MODEL_2_VARIABLE_5"] = property_type
    frame["MODEL_2_VARIABLE_3"] = _inject_missing(
        rng, frame["MODEL_2_VARIABLE_3"].to_numpy(), 0.06
    )

    # ==================================================================
    # MODEL_3 -- business loan scorecard.  Zero-count grade + unseen level
    # ==================================================================
    turnover = rng.lognormal(14.2, 0.8, n)
    business_vintage = np.clip(rng.gamma(2.5, 3.0, n), 0, 40)
    current_ratio = np.clip(rng.normal(1.45, 0.45, n), 0.1, 5)
    gst_filings = rng.integers(0, 13, n)
    business_type = rng.choice(
        ["TRADING", "MANUFACTURING", "SERVICES", "RETAIL", "CONTRACTOR"], size=n,
        p=[0.30, 0.20, 0.26, 0.18, 0.06],
    ).astype(object)
    # scenario 6: a level that never existed in the reference period
    gig_mask = (login.to_numpy() >= np.datetime64("2025-07-01")) & (rng.random(n) < 0.08)
    business_type[gig_mask] = "GIG_ECONOMY"
    m3_latent = (
        -0.55 * (np.log(turnover) - 14.2)
        - 0.05 * (business_vintage - 7.5)
        - 0.70 * (current_ratio - 1.45)
        - 0.06 * (gst_filings - 6)
        + np.where(business_type == "MANUFACTURING", -0.25, 0.08)
        + rng.normal(0, 1.0, n)
    )
    latent["MODEL_3"] = m3_latent
    frame["MODEL_3_VARIABLE_1"] = np.round(turnover, 0)
    frame["MODEL_3_VARIABLE_2"] = business_type
    frame["MODEL_3_VARIABLE_3"] = np.round(current_ratio, 3)
    frame["MODEL_3_VARIABLE_4"] = gst_filings.astype("float64")
    frame["MODEL_3_VARIABLE_2"] = _inject_missing(
        rng, frame["MODEL_3_VARIABLE_2"].to_numpy(object), 0.025
    )

    # ==================================================================
    # MODEL_4 -- newly implemented scorecard.  Thin data (scenario 12)
    # ==================================================================
    live_mask = login.to_numpy() >= np.datetime64("2025-07-01")
    scored_m4 = live_mask & (rng.random(n) < 0.16)
    deposit_ratio = np.clip(rng.normal(0.22, 0.10, n), 0, 1)
    app_velocity = rng.poisson(1.2, n).astype("float64")
    device_risk = rng.choice(["LOW", "MEDIUM", "HIGH"], size=n, p=[0.68, 0.25, 0.07])
    m4_latent = (
        -2.5 * (deposit_ratio - 0.22)
        + 0.35 * app_velocity
        + np.where(device_risk == "HIGH", 0.9, np.where(device_risk == "MEDIUM", 0.3, -0.15))
        + rng.normal(0, 1.0, n)
    )
    latent["MODEL_4"] = m4_latent
    frame["MODEL_4_VARIABLE_1"] = np.where(scored_m4, np.round(deposit_ratio, 4), np.nan)
    frame["MODEL_4_VARIABLE_2"] = np.where(scored_m4, app_velocity, np.nan)
    frame["MODEL_4_VARIABLE_3"] = np.where(scored_m4, device_risk, None)

    # ==================================================================
    # Segments, scores and subscores
    # ==================================================================
    reference_windows = {
        "MODEL_1": ("2024-01-01", "2024-06-30"),
        "MODEL_2": ("2024-01-01", "2024-12-31"),
        "MODEL_3": ("2024-07-01", "2024-12-31"),
        "MODEL_4": ("2025-07-01", "2025-09-30"),
    }
    login_array = login.to_numpy()
    for model, (window_start, window_end) in reference_windows.items():
        values = latent[model]
        in_window = (
            (login_array >= np.datetime64(window_start))
            & (login_array <= np.datetime64(window_end))
        )
        if model == "MODEL_4":
            in_window &= scored_m4
        sample = values[in_window]
        if sample.size < 50:                      # fall back if the window is tiny
            sample = values
        cutoffs = _cutoffs(sample)
        segments = _assign_segments(values, cutoffs)

        # final score: a 300-900 style scorecard score, monotone in -latent
        score = np.clip(
            np.round(650 - 42 * (values - np.nanmean(sample)) / max(1e-9, np.nanstd(sample))),
            300, 900,
        ).astype("float64")

        if model == "MODEL_4":
            segments = np.where(scored_m4, segments, None)
            score = np.where(scored_m4, score, np.nan)

        if model == "MODEL_3":
            # scenario 3: the scoring engine stops emitting the riskiest grade
            stopped = login_array >= np.datetime64("2025-07-01")
            riskiest = SEGMENT_ORDER[-1]
            second = SEGMENT_ORDER[-2]
            segments = np.where(stopped & (segments == riskiest), second, segments)

        frame[f"{model}_SEGMENT"] = segments
        frame[f"{model}_TOTAL_SCORE"] = score

    # variable-level subscores: points contribution, monotone in the characteristic
    subscore_specs = {
        "MODEL_1_VARIABLE_1": (lambda v: np.clip((v - 300) / 600 * 60, 0, 60)),
        "MODEL_1_VARIABLE_2": (lambda v: np.clip(45 - v * 0.45, 0, 45)),
        "MODEL_1_VARIABLE_3": (lambda v: np.clip(v * 0.18, 0, 40)),
        "MODEL_1_VARIABLE_4": (lambda v: np.clip(30 - v * 6.0, 0, 30)),
        "MODEL_2_VARIABLE_1": (lambda v: np.clip(55 - (v - 10) * 0.6, 0, 55)),
        "MODEL_2_VARIABLE_4": (lambda v: np.clip(40 - v * 30, 0, 40)),
        "MODEL_3_VARIABLE_1": (lambda v: np.clip((np.log(np.maximum(v, 1)) - 11) * 9, 0, 50)),
        "MODEL_3_VARIABLE_3": (lambda v: np.clip(v * 18, 0, 45)),
        "MODEL_4_VARIABLE_1": (lambda v: np.clip(v * 120, 0, 35)),
    }
    for column, transform in subscore_specs.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        frame[f"{column}_SUBSCORE"] = np.round(transform(values), 1)

    # ==================================================================
    # Scenario 4 -- missing and invalid segment values
    # ==================================================================
    m1_segment = frame["MODEL_1_SEGMENT"].to_numpy(object, copy=True)
    m1_segment[rng.random(n) < 0.015] = None
    invalid_draw = rng.random(n) < 0.004
    m1_segment[invalid_draw] = rng.choice(["CAT F", "UNKNOWN", "NA"], size=int(invalid_draw.sum()))
    frame["MODEL_1_SEGMENT"] = m1_segment

    m4_segment = frame["MODEL_4_SEGMENT"].to_numpy(object, copy=True)
    drop_m4 = scored_m4 & (rng.random(n) < 0.05)
    m4_segment[drop_m4] = None
    bad_m4 = scored_m4 & (rng.random(n) < 0.012)
    m4_segment[bad_m4] = "CAT X"
    frame["MODEL_4_SEGMENT"] = m4_segment

    # ==================================================================
    # Outcome: default flag, expected PD, right-censoring
    # ==================================================================
    blended = (
        0.55 * latent["MODEL_1"]
        + 0.25 * latent["MODEL_2"]
        + 0.20 * latent["MODEL_3"]
    )
    probability = 1.0 / (1.0 + np.exp(-(-2.75 + 0.55 * blended)))
    eventual_bad = rng.random(n) < probability

    window_end = pd.to_datetime(settings.end_date)
    months_on_book = (
        (window_end - login).dt.days / 30.44
    ).to_numpy()
    matured = months_on_book >= settings.outcome_window_months
    # a bad surfaces at a random point inside the window, so a partially
    # seasoned cohort shows only some of its eventual bads
    emergence = rng.uniform(2.0, settings.outcome_window_months, n)
    observed_bad = eventual_bad & (months_on_book >= emergence)
    frame["DEFAULT_FLAG"] = observed_bad.astype("int8")
    frame["PERFORMANCE_MATURED"] = matured.astype("int8")

    # expected PD, deliberately under-predicting in the later (drifted) period
    expected = probability * np.where(t > 0.6, 0.72, 1.0) * rng.uniform(0.9, 1.1, n)
    frame["EXPECTED_PD"] = np.round(np.clip(expected, 0.0005, 0.95), 5)

    # ==================================================================
    # Scenario 11 -- duplicate records
    # ==================================================================
    n_duplicates = max(1, int(n * settings.duplicate_lan_share))
    duplicate_rows = frame.sample(n=n_duplicates, random_state=settings.seed).copy()
    frame = pd.concat([frame, duplicate_rows], ignore_index=True)
    # a few genuine customer+date fan-outs with a different LAN
    fanout = frame.sample(n=max(1, n_duplicates // 2), random_state=settings.seed + 1).copy()
    fanout["LAN_ID"] = [f"LANDUP{idx:06d}" for idx in range(len(fanout))]
    frame = pd.concat([frame, fanout], ignore_index=True)

    frame = frame.sample(frac=1.0, random_state=settings.seed + 2).reset_index(drop=True)
    frame["LOGIN_DATE"] = pd.to_datetime(frame["LOGIN_DATE"]).dt.date
    ordered = [
        "LAN_ID", "CUSTOMER_ID", "LOGIN_DATE", "PRODUCT", "CHANNEL", "BRANCH_STATE",
        "SANCTION_AMOUNT",
    ]
    for model in ("MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"):
        ordered.append(f"{model}_SEGMENT")
        ordered.append(f"{model}_TOTAL_SCORE")
        for column in sorted(c for c in frame.columns if c.startswith(f"{model}_VARIABLE")):
            if column not in ordered:
                ordered.append(column)
    ordered.extend(["DEFAULT_FLAG", "PERFORMANCE_MATURED", "EXPECTED_PD"])
    ordered = [c for c in ordered if c in frame.columns]
    remaining = [c for c in frame.columns if c not in ordered]
    return frame[ordered + remaining]


def scenario_frame() -> pd.DataFrame:
    """The planted-scenario manifest, written into the Excel pack."""
    return pd.DataFrame(SCENARIOS, columns=["ID", "SCENARIO", "WHERE", "EXPECT"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic NBFC scorecard extract for the monitoring pipeline."
    )
    parser.add_argument("--rows", type=int, default=25_000, help="number of applications")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="data/synthetic_scorecard_data.csv",
        help="output path (.csv or .parquet)",
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-09-20",
                        help="deliberately mid-quarter so the last quarter is incomplete")
    arguments = parser.parse_args(argv)

    settings = GeneratorConfig(
        n_rows=arguments.rows,
        seed=arguments.seed,
        start_date=datetime.strptime(arguments.start, "%Y-%m-%d").date(),
        end_date=datetime.strptime(arguments.end, "%Y-%m-%d").date(),
    )
    frame = generate_synthetic_dataset(settings)
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)

    print(f"Wrote {len(frame):,} rows x {frame.shape[1]} columns to {out}")
    print(f"Date span: {frame['LOGIN_DATE'].min()} to {frame['LOGIN_DATE'].max()}")
    print("\nSegment distribution by model:")
    for model in ("MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"):
        column = f"{model}_SEGMENT"
        counts = frame[column].value_counts(dropna=False)
        print(f"  {model}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    print("\nPlanted scenarios:")
    for scenario in SCENARIOS:
        print(f"  {scenario['ID']:>2}. {scenario['SCENARIO']} [{scenario['WHERE']}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GeneratorConfig", "generate_synthetic_dataset", "scenario_frame", "SCENARIOS", "main",
]
