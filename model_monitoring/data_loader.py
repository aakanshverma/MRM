"""Dataset loading with column pruning, chunking and dtype optimisation.

Wide scorecard extracts are the norm (hundreds to thousands of columns), so the
loader is built around one principle: **read only what the configuration asks
for**.  The header is inspected first, variable discovery is resolved against
it, and only the resulting column list is materialised.  On a 3,000-column
extract where a run monitors 200 characteristics this alone cuts memory by an
order of magnitude.

If the pruned frame still does not fit in memory the loader says so explicitly
(with the measured footprint and concrete remedies) rather than silently
downsampling the data.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DataSourceConfig, MonitoringConfig
from .logging_config import ExecutionLog, get_logger
from .utils import pct

logger = get_logger("data_loader")


class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be read or is unusable."""


@dataclass
class LoadResult:
    """The loaded frame plus provenance for the governance sheet."""

    frame: pd.DataFrame
    source_path: str
    source_format: str
    rows_read: int
    columns_available: List[str]
    columns_loaded: List[str]
    columns_pruned: int
    memory_mb: float
    file_size_mb: Optional[float] = None
    chunks_read: int = 1
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.rows_read:,} rows x {len(self.columns_loaded)} columns "
            f"({self.memory_mb:.1f} MB in memory; {self.columns_pruned} of "
            f"{len(self.columns_available)} source columns pruned) from "
            f"{self.source_format.upper()} '{self.source_path}'"
        )

    def as_rows(self) -> List[Dict[str, Any]]:
        rows = [
            {"ITEM": "Input file", "VALUE": self.source_path},
            {"ITEM": "Input format", "VALUE": self.source_format.upper()},
            {"ITEM": "File size (MB)",
             "VALUE": "n/a" if self.file_size_mb is None else round(self.file_size_mb, 2)},
            {"ITEM": "Rows read", "VALUE": self.rows_read},
            {"ITEM": "Columns available in source", "VALUE": len(self.columns_available)},
            {"ITEM": "Columns loaded", "VALUE": len(self.columns_loaded)},
            {"ITEM": "Columns pruned (not required by config)", "VALUE": self.columns_pruned},
            {"ITEM": "In-memory footprint (MB)", "VALUE": round(self.memory_mb, 2)},
            {"ITEM": "Chunks read", "VALUE": self.chunks_read},
        ]
        rows.extend({"ITEM": "Load note", "VALUE": note} for note in self.notes)
        return rows


# ---------------------------------------------------------------------------
# Header inspection
# ---------------------------------------------------------------------------


def peek_columns(source: DataSourceConfig) -> List[str]:
    """Read only the header of the configured source.

    This is what makes column pruning possible: discovery patterns need the
    column universe, but we do not want to pay for loading it.
    """
    fmt = source.resolved_format()
    if fmt == "dataframe":
        raise DataLoadError(
            "peek_columns() needs a file-backed source; pass the DataFrame directly instead."
        )
    path = _require_path(source)
    try:
        if fmt == "csv":
            head = pd.read_csv(
                path, nrows=0, sep=source.delimiter, encoding=source.encoding,
                engine="python",
            )
            return [str(c) for c in head.columns]
        if fmt == "parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                return [str(c) for c in pq.ParquetFile(path).schema.names]
            except ImportError:
                return [str(c) for c in pd.read_parquet(path).columns]
        if fmt == "excel":
            head = pd.read_excel(path, sheet_name=source.sheet_name, nrows=0)
            if isinstance(head, dict):  # sheet_name=None returns a dict
                head = next(iter(head.values()))
            return [str(c) for c in head.columns]
    except FileNotFoundError as exc:
        raise DataLoadError(f"Input file not found: {path}") from exc
    except Exception as exc:
        raise DataLoadError(f"Could not read the header of {path}: {exc}") from exc
    raise DataLoadError(f"Unsupported source format {fmt!r}.")


def _require_path(source: DataSourceConfig) -> Path:
    if not source.path:
        raise DataLoadError("data_source.path is not set.")
    path = Path(source.path).expanduser()
    if not path.exists():
        raise DataLoadError(
            f"Input file not found: {path}. Generate the synthetic sample with "
            "'python -m model_monitoring.synthetic_data' or point data_source.path at your extract."
        )
    return path


# ---------------------------------------------------------------------------
# Dtype optimisation
# ---------------------------------------------------------------------------


def optimise_dtypes(
    frame: pd.DataFrame,
    *,
    exclude: Sequence[str] = (),
    category_max_ratio: float = 0.5,
    category_max_unique: int = 1_000,
) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality text to ``category``.

    Operates in place where pandas allows it.  Columns in ``exclude`` (keys,
    dates) are left untouched so downstream joins and date maths stay exact.
    """
    excluded = set(exclude)
    n_rows = max(1, len(frame))
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        try:
            if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            if pd.api.types.is_integer_dtype(series):
                frame[column] = pd.to_numeric(series, downcast="integer")
            elif pd.api.types.is_float_dtype(series):
                frame[column] = pd.to_numeric(series, downcast="float")
            else:
                n_unique = int(series.nunique(dropna=True))
                if n_unique <= category_max_unique and (n_unique / n_rows) <= category_max_ratio:
                    frame[column] = series.astype("category")
        except Exception as exc:  # pragma: no cover - never fail a run on this
            logger.debug("dtype optimisation skipped for %s: %s", column, exc)
    return frame


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(
    config: MonitoringConfig,
    *,
    frame: Optional[pd.DataFrame] = None,
    execution_log: Optional[ExecutionLog] = None,
    resolve_variables: bool = True,
) -> LoadResult:
    """Load the configured dataset and resolve each model's variable list.

    When ``frame`` is supplied it is used as-is (the in-memory / library entry
    point); otherwise the configured file is read.  Variable discovery runs
    against the *available* column universe, then the frame is pruned to the
    required columns.
    """
    source = config.data_source
    fmt = "dataframe" if frame is not None else source.resolved_format()
    notes: List[str] = []
    file_size_mb: Optional[float] = None

    if frame is not None:
        available = [str(c) for c in frame.columns]
        path_label = "<in-memory DataFrame>"
    else:
        path = _require_path(source)
        path_label = str(path)
        file_size_mb = os.path.getsize(path) / (1024 ** 2)
        available = peek_columns(source)
        logger.info(
            "Source '%s' exposes %d columns (%.1f MB on disk).",
            path_label, len(available), file_size_mb,
        )

    # ---- resolve the variable universe before reading the body -----------
    if resolve_variables:
        dtype_hints = _sample_dtype_hints(source, available) if frame is None else None
        if frame is not None:
            from .utils import infer_dtype_hints

            dtype_hints = infer_dtype_hints(frame)
        warnings = config.resolve_variables(available, dtype_hints=dtype_hints, logger=logger)
        if execution_log is not None:
            for message in warnings:
                execution_log.warning(message)

    required = config.required_columns()
    missing_required = [c for c in required if c not in set(available)]
    usable = [c for c in required if c in set(available)]
    if missing_required:
        notes.append(
            f"{len(missing_required)} configured column(s) absent from the source: "
            + ", ".join(missing_required[:15])
            + (" ..." if len(missing_required) > 15 else "")
        )
        logger.warning(notes[-1])
        if execution_log is not None:
            execution_log.warning(notes[-1])

    prune = source.prune_columns and bool(usable)
    columns_to_load = usable if prune else available

    if frame is not None:
        loaded = frame.loc[:, [c for c in columns_to_load if c in frame.columns]].copy()
        chunks_read = 1
    else:
        loaded, chunks_read = _read_file(source, fmt, columns_to_load if prune else None, notes)

    if loaded.empty:
        raise DataLoadError(
            f"The dataset read from {path_label} is empty. Check the file, the delimiter "
            f"({source.delimiter!r}) and any nrows limit."
        )

    # ---- types -----------------------------------------------------------
    key_columns = [config.lan_column, config.customer_column, config.date_column]
    loaded = _coerce_dates(loaded, config, notes, execution_log)
    if source.optimise_dtypes:
        loaded = optimise_dtypes(loaded, exclude=key_columns)

    memory_mb = float(loaded.memory_usage(deep=True).sum()) / (1024 ** 2)
    result = LoadResult(
        frame=loaded,
        source_path=path_label,
        source_format=fmt,
        rows_read=int(len(loaded)),
        columns_available=available,
        columns_loaded=[str(c) for c in loaded.columns],
        columns_pruned=max(0, len(available) - len(loaded.columns)),
        memory_mb=memory_mb,
        file_size_mb=file_size_mb,
        chunks_read=chunks_read,
        notes=notes,
    )
    logger.info("Loaded %s", result.summary())
    _warn_on_memory(result, execution_log)
    gc.collect()
    return result


def _read_file(
    source: DataSourceConfig,
    fmt: str,
    columns: Optional[List[str]],
    notes: List[str],
) -> tuple[pd.DataFrame, int]:
    """Read the dataset body, honouring chunking for CSV."""
    path = _require_path(source)
    if fmt == "csv":
        read_kwargs: Dict[str, Any] = dict(
            sep=source.delimiter,
            encoding=source.encoding,
            na_values=source.na_values,
            keep_default_na=True,
        )
        if columns:
            read_kwargs["usecols"] = columns
        if source.nrows:
            read_kwargs["nrows"] = source.nrows
            notes.append(f"nrows limit applied: {source.nrows:,} (explicitly configured).")
        if source.chunksize:
            frames: List[pd.DataFrame] = []
            chunks = 0
            for chunk in pd.read_csv(path, chunksize=source.chunksize, **read_kwargs):
                frames.append(optimise_dtypes(chunk) if source.optimise_dtypes else chunk)
                chunks += 1
            notes.append(f"Read in {chunks} chunk(s) of {source.chunksize:,} rows.")
            frame = pd.concat(frames, ignore_index=True, copy=False)
            del frames
            gc.collect()
            return frame, chunks
        return pd.read_csv(path, **read_kwargs), 1

    if fmt == "parquet":
        frame = pd.read_parquet(path, columns=columns)
        if source.nrows:
            frame = frame.head(source.nrows)
            notes.append(f"nrows limit applied: {source.nrows:,} (explicitly configured).")
        return frame, 1

    if fmt == "excel":
        frame = pd.read_excel(
            path, sheet_name=source.sheet_name, usecols=columns,
            nrows=source.nrows, na_values=source.na_values,
        )
        if isinstance(frame, dict):
            name, frame = next(iter(frame.items()))
            notes.append(f"Multiple sheets present; used '{name}'.")
        return frame, 1

    raise DataLoadError(f"Unsupported source format {fmt!r}.")


def _coerce_dates(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    notes: List[str],
    execution_log: Optional[ExecutionLog],
) -> pd.DataFrame:
    """Parse the login date (and any outcome/vintage date) to datetime64."""
    date_columns = [config.date_column]
    if config.outcome.vintage_date_column:
        date_columns.append(config.outcome.vintage_date_column)
    for column in date_columns:
        if column not in frame.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        before_missing = int(frame[column].isna().sum())
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        after_missing = int(frame[column].isna().sum())
        unparsed = after_missing - before_missing
        if unparsed > 0:
            message = (
                f"{unparsed:,} value(s) in '{column}' could not be parsed as a date and became "
                "NaT; those records cannot be assigned to a quarter."
            )
            notes.append(message)
            logger.warning(message)
            if execution_log is not None:
                execution_log.warning(message)
    return frame


def _sample_dtype_hints(
    source: DataSourceConfig, available: Sequence[str], sample_rows: int = 5_000
) -> Dict[str, str]:
    """Infer numeric/categorical hints from a small sample of the source.

    Only used to seed *discovered* variables; explicit configuration always wins.
    """
    from .utils import infer_dtype_hints

    fmt = source.resolved_format()
    try:
        path = _require_path(source)
        if fmt == "csv":
            sample = pd.read_csv(
                path, nrows=sample_rows, sep=source.delimiter, encoding=source.encoding,
                na_values=source.na_values,
            )
        elif fmt == "parquet":
            sample = pd.read_parquet(path).head(sample_rows)
        elif fmt == "excel":
            sample = pd.read_excel(path, sheet_name=source.sheet_name, nrows=sample_rows)
            if isinstance(sample, dict):
                sample = next(iter(sample.values()))
        else:  # pragma: no cover
            return {}
        return infer_dtype_hints(sample)
    except Exception as exc:  # pragma: no cover - hints are best-effort
        logger.debug("dtype hint sampling failed: %s", exc)
        return {}


def _warn_on_memory(result: LoadResult, execution_log: Optional[ExecutionLog]) -> None:
    """Explain the limitation (and the remedies) instead of downsampling."""
    threshold_mb = float(os.environ.get("MRM_MEMORY_WARN_MB", "2048"))
    if result.memory_mb < threshold_mb:
        return
    message = (
        f"The pruned dataset occupies {result.memory_mb:,.0f} MB in memory, above the "
        f"{threshold_mb:,.0f} MB advisory threshold. The pipeline will continue but may be "
        "slow or exhaust RAM. Memory-efficient options (none of which drop data): "
        "(1) set data_source.chunksize to stream the CSV, (2) convert the extract to Parquet "
        "so only the required columns are read from disk, (3) restrict "
        "runtime.csi_variable_whitelist and run the remaining characteristics in a second pass, "
        "(4) run one model per execution. See docs/TROUBLESHOOTING.md."
    )
    logger.warning(message)
    result.notes.append(message)
    if execution_log is not None:
        execution_log.warning(message)


def join_outcome_source(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
) -> pd.DataFrame:
    """Left-join a separately supplied outcome file onto the extract.

    Bureau-derived flags usually arrive as their own file from a later pull --
    one row per customer rather than a column already on the application
    extract.  The join is always a LEFT join: an application with no matching
    outcome row keeps a null flag and is excluded from performance metrics, it
    is never dropped from the stability analysis and never defaulted to good.

    Diagnostics that matter are logged rather than left to be discovered: the
    match rate, duplicate keys in the outcome file (which would fan out the
    extract), and keys present in the outcome file but absent from the extract.
    """
    source = config.outcome.source
    if source is None or not source.enabled:
        return frame

    join_on = source.join_on or config.customer_column
    key_column = source.key_column or join_on
    if join_on not in frame.columns:
        message = (
            f"Outcome file cannot be joined: '{join_on}' is not in the extract. "
            "Set outcome.source.join_on to a column that is."
        )
        logger.error(message)
        if execution_log is not None:
            execution_log.error(message)
        raise DataLoadError(message)

    path = Path(source.path).expanduser()
    if not path.exists():
        raise DataLoadError(f"Outcome file not found: {path}")

    fmt = source.resolved_format()
    wanted = None
    if source.columns:
        wanted = list(dict.fromkeys([key_column, *source.columns]))
    try:
        if fmt == "csv":
            outcome = pd.read_csv(
                path, sep=source.delimiter, encoding=source.encoding, usecols=wanted
            )
        elif fmt == "parquet":
            outcome = pd.read_parquet(path, columns=wanted)
        else:
            outcome = pd.read_excel(path, sheet_name=source.sheet_name, usecols=wanted)
            if isinstance(outcome, dict):
                outcome = next(iter(outcome.values()))
    except Exception as exc:
        raise DataLoadError(f"Could not read the outcome file {path}: {exc}") from exc

    if key_column not in outcome.columns:
        raise DataLoadError(
            f"Outcome file {path} has no key column '{key_column}'. Available: "
            f"{list(outcome.columns)[:20]}"
        )

    # Duplicate keys would fan out the extract and silently multiply records.
    duplicates = int(outcome[key_column].duplicated().sum())
    if duplicates:
        message = (
            f"The outcome file has {duplicates:,} duplicated '{key_column}' value(s). Keeping "
            "the first row per key to avoid fanning out the application extract; confirm the "
            "file is one row per key."
        )
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
        outcome = outcome.drop_duplicates(subset=[key_column], keep="first")

    overlapping = [
        column for column in outcome.columns
        if column != key_column and column in frame.columns
    ]
    if overlapping:
        message = (
            f"Outcome file column(s) {overlapping} already exist in the extract and are taken "
            "from the outcome file."
        )
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
        frame = frame.drop(columns=overlapping)

    before = len(frame)
    merged = frame.merge(
        outcome.rename(columns={key_column: join_on}), on=join_on, how="left"
    )
    if len(merged) != before:  # pragma: no cover - guarded by the de-duplication above
        raise DataLoadError(
            f"Joining the outcome file changed the row count from {before:,} to "
            f"{len(merged):,}. The outcome file must be one row per '{key_column}'."
        )

    outcome_column = config.outcome.outcome_column
    if outcome_column and outcome_column in merged.columns:
        matched = int(merged[outcome_column].notna().sum())
        rate = pct(matched, before, 0.0) or 0.0
        message = (
            f"Outcome file joined on '{join_on}': {matched:,} of {before:,} application(s) "
            f"matched an outcome row ({rate:.1f}%). Unmatched applications keep a null flag and "
            "are excluded from performance metrics, never counted as good."
        )
        logger.info(message)
        if execution_log is not None:
            execution_log.info(message)
        if rate < 50.0:
            warning = (
                f"Only {rate:.1f}% of applications matched an outcome row. Check that "
                f"'{join_on}' and '{key_column}' are the same identifier on the same format "
                "(leading zeros, prefixes and type coercion are the usual culprits) before "
                "reading any performance figure in this pack."
            )
            logger.warning(warning)
            if execution_log is not None:
                execution_log.warning(warning)
    else:
        message = (
            f"The outcome file was joined but '{outcome_column}' is not among its columns; "
            "performance metrics will remain unavailable."
        )
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
    return merged


def iter_source_chunks(
    source: DataSourceConfig, columns: Optional[Sequence[str]] = None, chunksize: int = 250_000
) -> Iterable[pd.DataFrame]:
    """Stream a CSV/Parquet source in chunks.

    Exposed for out-of-core pre-aggregation on datasets that genuinely do not
    fit in memory (see docs/TROUBLESHOOTING.md).
    """
    fmt = source.resolved_format()
    path = _require_path(source)
    if fmt == "csv":
        yield from pd.read_csv(
            path, chunksize=chunksize, sep=source.delimiter, encoding=source.encoding,
            usecols=list(columns) if columns else None, na_values=source.na_values,
        )
    elif fmt == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=chunksize, columns=list(columns) if columns else None
            ):
                yield batch.to_pandas()
        except ImportError:
            yield pd.read_parquet(path, columns=list(columns) if columns else None)
    else:
        raise DataLoadError(f"Chunked iteration is not supported for format {fmt!r}.")


__all__ = [
    "DataLoadError", "LoadResult", "load_dataset", "peek_columns",
    "optimise_dtypes", "iter_source_chunks", "join_outcome_source",
]
