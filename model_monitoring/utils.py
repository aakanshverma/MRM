"""Small shared helpers: quarter handling, safe maths, dtype probing.

Nothing here encodes a business rule -- those live in
:mod:`model_monitoring.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NOT_AVAILABLE = "NOT AVAILABLE"

MISSING_BIN = "MISSING"
OTHER_BIN = "OTHER"
UNSEEN_BIN = "UNSEEN"
#: Bins that do not come from the reference distribution of real values.
SPECIAL_BINS = (MISSING_BIN, OTHER_BIN, UNSEEN_BIN)


# ---------------------------------------------------------------------------
# Quarter handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuarterInfo:
    """A monitoring quarter derived from ``LOGIN_DATE``."""

    label: str                 # e.g. "2024Q3"
    period_start: date
    period_end: date
    record_count: int
    is_complete: bool
    #: First / last observed date inside the quarter (data coverage).
    observed_start: Optional[date] = None
    observed_end: Optional[date] = None
    completeness_note: str = ""

    @property
    def sort_key(self) -> Tuple[int, int]:
        year, quarter = self.label.split("Q")
        return int(year), int(quarter)

    def describe(self) -> str:
        state = "complete" if self.is_complete else "INCOMPLETE"
        return f"{self.label} ({self.period_start} to {self.period_end}, {state}, n={self.record_count:,})"


def to_quarter_series(dates: pd.Series) -> pd.Series:
    """Map a datetime series to quarter labels like ``2024Q3`` (NaT -> NA)."""
    parsed = pd.to_datetime(dates, errors="coerce")
    periods = parsed.dt.to_period("Q")
    return periods.astype("string")


def build_quarters(
    dates: pd.Series,
    *,
    data_max_date: Optional[pd.Timestamp] = None,
    data_min_date: Optional[pd.Timestamp] = None,
    min_records: int = 0,
) -> List[QuarterInfo]:
    """Derive the monitoring quarters present in ``dates``.

    A quarter is **complete** when the dataset covers its full calendar span:
    the data must extend to (or beyond) the quarter end, and must start at (or
    before) the quarter start.  Everything else is flagged ``INCOMPLETE`` and
    is excluded from the primary comparison unless the user opts in.
    """
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty:
        return []
    max_date = pd.Timestamp(data_max_date) if data_max_date is not None else parsed.max()
    min_date = pd.Timestamp(data_min_date) if data_min_date is not None else parsed.min()

    periods = parsed.dt.to_period("Q")
    quarters: List[QuarterInfo] = []
    for period, group in parsed.groupby(periods, observed=True):
        start = period.start_time
        end = period.end_time.normalize()
        notes: List[str] = []
        complete = True
        if max_date < end:
            complete = False
            notes.append(
                f"dataset ends {max_date.date()} before quarter end {end.date()}"
            )
        if min_date > start:
            complete = False
            notes.append(
                f"dataset starts {min_date.date()} after quarter start {start.date()}"
            )
        count = int(group.shape[0])
        if min_records and count < min_records:
            notes.append(f"only {count} records (< min_quarter_records={min_records})")
        quarters.append(
            QuarterInfo(
                label=str(period),
                period_start=start.date(),
                period_end=end.date(),
                record_count=count,
                is_complete=complete,
                observed_start=group.min().date(),
                observed_end=group.max().date(),
                completeness_note="; ".join(notes),
            )
        )
    quarters.sort(key=lambda q: q.sort_key)
    return quarters


def periods_overlap(
    start_a: Optional[date], end_a: Optional[date],
    start_b: Optional[date], end_b: Optional[date],
) -> bool:
    """``True`` when two closed date ranges share at least one day."""
    if None in (start_a, end_a, start_b, end_b):
        return False
    return start_a <= end_b and start_b <= end_a


def previous_quarter_label(label: str) -> str:
    """``2024Q1`` -> ``2023Q4``."""
    year, quarter = label.split("Q")
    year, quarter = int(year), int(quarter)
    if quarter == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{quarter - 1}"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def safe_divide(numerator: Any, denominator: Any, default: Optional[float] = None) -> Optional[float]:
    """Division that returns ``default`` instead of raising / returning inf."""
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return default
    if den == 0 or den != den or num != num:
        return default
    return num / den


def pct(numerator: Any, denominator: Any, default: Optional[float] = None) -> Optional[float]:
    """Percentage (0-100) with safe division."""
    value = safe_divide(numerator, denominator, None)
    return default if value is None else value * 100.0


def round_or_none(value: Any, decimals: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return round(numeric, decimals)


def is_missing(value: Any) -> bool:
    """``True`` for None/NaN/NaT/pd.NA and for blank strings."""
    try:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
        return bool(np.all(result))
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# dtype probing
# ---------------------------------------------------------------------------


def is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not bool(
        pd.api.types.is_bool_dtype(series)
    )


def infer_variable_kind(series: pd.Series, categorical_max_unique: int = 15) -> str:
    """Return ``'numeric'`` or ``'categorical'`` for an unlabelled column.

    A numeric column with very few distinct values (flags, bands, codes) is
    treated as categorical: quantile binning such a column produces degenerate
    bins and a meaningless CSI.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "numeric"
    if is_numeric(series):
        try:
            n_unique = int(series.nunique(dropna=True))
        except Exception:  # pragma: no cover
            return "numeric"
        return "categorical" if n_unique <= categorical_max_unique else "numeric"
    return "categorical"


def infer_dtype_hints(
    frame: pd.DataFrame, categorical_max_unique: int = 15
) -> Dict[str, str]:
    """``{column: 'numeric'|'categorical'}`` for every column of ``frame``."""
    return {
        str(column): infer_variable_kind(frame[column], categorical_max_unique)
        for column in frame.columns
    }


def count_infinite(series: pd.Series) -> int:
    """Number of +/-inf values (0 for non-numeric columns)."""
    if not is_numeric(series):
        return 0
    try:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        return 0
    return int(np.isinf(values).sum())


def memory_usage_mb(frame: pd.DataFrame) -> float:
    try:
        return float(frame.memory_usage(deep=True).sum()) / (1024 ** 2)
    except Exception:  # pragma: no cover
        return float("nan")


# ---------------------------------------------------------------------------
# Formatting helpers used by several report builders
# ---------------------------------------------------------------------------


def format_count(value: Any) -> str:
    if is_missing(value):
        return NOT_AVAILABLE
    return f"{int(value):,}"


def format_pct(value: Any, decimals: int = 2) -> str:
    if is_missing(value):
        return NOT_AVAILABLE
    return f"{float(value):.{decimals}f}%"


def truncate_text(text: Any, limit: int = 32_000) -> str:
    """Excel rejects cell values beyond ~32,767 characters."""
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def chunked(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    """Yield ``items`` in lists of at most ``size`` elements."""
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    return list(dict.fromkeys(items))


__all__ = [
    "NOT_AVAILABLE", "MISSING_BIN", "OTHER_BIN", "UNSEEN_BIN", "SPECIAL_BINS",
    "QuarterInfo", "to_quarter_series", "build_quarters", "previous_quarter_label",
    "periods_overlap",
    "safe_divide", "pct", "round_or_none", "is_missing", "is_numeric",
    "infer_variable_kind", "infer_dtype_hints", "count_infinite", "memory_usage_mb",
    "format_count", "format_pct", "truncate_text", "chunked", "dedupe_preserve_order",
]
