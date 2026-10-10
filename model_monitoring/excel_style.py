"""Excel presentation toolkit: theme, table writer, conditional formatting.

Strictly presentation -- no business logic.  The analytical modules produce
DataFrames; this module decides what they look like.  That separation is what
lets the same numbers be rendered into a different template (or a different
format entirely) without touching a single calculation.

Design rules enforced here
--------------------------
* Colour is never the only signal: every conditionally-coloured cell carries a
  text label (``RED`` / ``AMBER`` / ``GREEN`` / ``GREY``, or a ``1-LOWEST RISK``
  style band), so the pack survives printing in greyscale and is readable to a
  colour-blind reader.
* The risk palette is derived from the *configured* risk ranks, so reversing
  ``risk_direction`` reverses the colours instead of inverting their meaning.
* Formatting cost is bounded: per-cell styling is applied up to a configurable
  row limit, above which column-level formats are used so a 200,000-row detail
  sheet still opens.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, DataBarRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

NAVY = "1F3864"
NAVY_LIGHT = "2F5597"
BAND = "D9E1F2"
BAND_DARK = "B4C6E7"
WHITE = "FFFFFF"
INK = "203864"
MUTED = "595959"
RULE = "BFBFBF"

SEVERITY_FILL: Dict[str, str] = {
    "GREEN": "C6EFCE",
    "AMBER": "FFEB9C",
    "RED": "FFC7CE",
    "GREY": "DCDCDC",
}
SEVERITY_FONT: Dict[str, str] = {
    "GREEN": "006100",
    "AMBER": "9C5700",
    "RED": "9C0006",
    "GREY": "3F3F3F",
}

#: Light-fill risk ramp, safest -> riskiest.  Used for ordered risk segments.
RISK_RAMP: List[str] = ["C6EFCE", "DDEBC2", "FFEB9C", "FCD9B4", "F8C7B0", "FFC7CE"]

FORMAT_INDEX = "0.0000"
FORMAT_INDEX_SHORT = "0.000"
FORMAT_PCT = '0.00"%"'
FORMAT_PCT_SIGNED = '+0.00"%";-0.00"%";0.00"%"'
FORMAT_PP_SIGNED = '+0.00" pp";-0.00" pp";0.00" pp"'
FORMAT_COUNT = "#,##0"
FORMAT_PROPORTION = "0.0000"
FORMAT_RATIO = "0.000"
FORMAT_DATE = "yyyy-mm-dd"

THIN = Side(style="thin", color=RULE)
BORDER_ALL = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BORDER_BOTTOM = Border(bottom=Side(style="medium", color=NAVY))


def sanitise_sheet_name(name: str) -> str:
    """Excel sheet names: <=31 chars and none of ``[]:*?/\\``."""
    cleaned = re.sub(r"[\[\]:*?/\\]", "-", str(name)).strip()
    return cleaned[:31] or "Sheet"


def sanitise_table_name(name: str) -> str:
    """Excel table names must be unique, alphanumeric/underscore, not cell-like."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"t_{cleaned}"
    return cleaned[:250]


# ---------------------------------------------------------------------------
# Risk palette
# ---------------------------------------------------------------------------


def risk_fill_for_rank(rank: Optional[int], n_segments: int) -> Optional[PatternFill]:
    """Fill colour for an ordered risk segment, honouring the configured direction.

    ``rank`` is always "higher = riskier" (see ``SegmentConfig.risk_rank``), so a
    reversed ``risk_direction`` produces reversed ranks and therefore reversed
    colours -- the palette can never imply that the riskiest grade is safe.
    """
    if rank is None or n_segments < 2:
        return None
    position = (int(rank) - 1) / (n_segments - 1)
    index = min(len(RISK_RAMP) - 1, max(0, round(position * (len(RISK_RAMP) - 1))))
    return PatternFill(start_color=RISK_RAMP[index], end_color=RISK_RAMP[index], fill_type="solid")


def severity_fill(severity: Any) -> Optional[PatternFill]:
    key = str(severity).strip().upper()
    colour = SEVERITY_FILL.get(key)
    if colour is None:
        return None
    return PatternFill(start_color=colour, end_color=colour, fill_type="solid")


def severity_font(severity: Any, bold: bool = True) -> Optional[Font]:
    key = str(severity).strip().upper()
    colour = SEVERITY_FONT.get(key)
    if colour is None:
        return None
    return Font(color=colour, bold=bold, size=10)


# ---------------------------------------------------------------------------
# Title / section blocks
# ---------------------------------------------------------------------------


def write_title(
    ws: Worksheet,
    title: str,
    subtitle: Optional[str] = None,
    *,
    row: int = 1,
    width: int = 12,
    note: Optional[str] = None,
) -> int:
    """Dark title banner; returns the next free row."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(2, width))
    cell = ws.cell(row=row, column=1, value=str(title))
    cell.font = Font(bold=True, size=15, color=WHITE)
    cell.fill = PatternFill(start_color=NAVY, end_color=NAVY, fill_type="solid")
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[row].height = 26
    current = row + 1

    if subtitle:
        ws.merge_cells(start_row=current, start_column=1, end_row=current, end_column=max(2, width))
        cell = ws.cell(row=current, column=1, value=str(subtitle))
        cell.font = Font(bold=False, size=10, color=WHITE)
        cell.fill = PatternFill(start_color=NAVY_LIGHT, end_color=NAVY_LIGHT, fill_type="solid")
        cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[current].height = 18
        current += 1

    if note:
        ws.merge_cells(start_row=current, start_column=1, end_row=current, end_column=max(2, width))
        cell = ws.cell(row=current, column=1, value=str(note))
        cell.font = Font(italic=True, size=9, color=MUTED)
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False, indent=1)
        current += 1
    return current + 1


def write_section(
    ws: Worksheet, row: int, text: str, *, width: int = 12, note: Optional[str] = None
) -> int:
    """Banded section header; returns the next free row."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(2, width))
    cell = ws.cell(row=row, column=1, value=str(text))
    cell.font = Font(bold=True, size=11, color=INK)
    cell.fill = PatternFill(start_color=BAND, end_color=BAND, fill_type="solid")
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[row].height = 20
    current = row + 1
    if note:
        ws.merge_cells(start_row=current, start_column=1, end_row=current, end_column=max(2, width))
        cell = ws.cell(row=current, column=1, value=str(note))
        cell.font = Font(italic=True, size=9, color=MUTED)
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True, indent=1)
        ws.row_dimensions[current].height = 14 * max(1, (len(str(note)) // (width * 14) + 1))
        current += 1
    return current


def write_note(ws: Worksheet, row: int, text: str, *, width: int = 12) -> int:
    """Wrapped explanatory paragraph; returns the next free row."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(2, width))
    cell = ws.cell(row=row, column=1, value=str(text))
    cell.font = Font(italic=True, size=9, color=MUTED)
    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True, indent=1)
    estimated_lines = max(1, len(str(text)) // max(40, width * 13) + 1)
    ws.row_dimensions[row].height = 12.5 * estimated_lines
    return row + 1


def write_key_values(
    ws: Worksheet,
    row: int,
    items: Sequence[Tuple[str, Any]],
    *,
    label_col: int = 1,
    value_col: int = 3,
    severities: Optional[Mapping[str, Any]] = None,
    label_width: int = 2,
) -> int:
    """Label/value rows, optionally severity-coloured by label."""
    for label, value in items:
        ws.merge_cells(
            start_row=row, start_column=label_col,
            end_row=row, end_column=label_col + max(0, label_width - 1),
        )
        label_cell = ws.cell(row=row, column=label_col, value=str(label))
        label_cell.font = Font(bold=True, size=10, color=INK)
        label_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        value_cell = ws.cell(row=row, column=value_col, value=value)
        value_cell.font = Font(size=10)
        value_cell.alignment = Alignment(horizontal="left", vertical="center")
        severity = (severities or {}).get(label)
        if severity is not None:
            fill = severity_fill(severity)
            font = severity_font(severity)
            if fill is not None:
                value_cell.fill = fill
            if font is not None:
                value_cell.font = font
        row += 1
    return row


# ---------------------------------------------------------------------------
# Table writing
# ---------------------------------------------------------------------------


def write_frame(
    ws: Worksheet,
    frame: pd.DataFrame,
    *,
    start_row: int = 1,
    start_col: int = 1,
    number_formats: Optional[Mapping[str, str]] = None,
    max_rows: Optional[int] = None,
    style_row_limit: int = 50_000,
    header_wrap: bool = True,
) -> Dict[str, Any]:
    """Write a DataFrame as a formatted block.

    Returns a dict with ``header_row``, ``first_data_row``, ``last_row``,
    ``last_col``, ``columns`` (name -> column index), ``rows_written`` and
    ``truncated``.
    """
    columns = [str(c) for c in frame.columns]
    header_row = start_row
    for offset, name in enumerate(columns):
        cell = ws.cell(row=header_row, column=start_col + offset, value=name)
        cell.font = Font(bold=True, size=9, color=WHITE)
        cell.fill = PatternFill(start_color=NAVY, end_color=NAVY, fill_type="solid")
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=header_wrap
        )
        cell.border = BORDER_ALL
    ws.row_dimensions[header_row].height = 30 if header_wrap else 16

    body = frame
    truncated = False
    if max_rows is not None and len(frame) > max_rows:
        body = frame.head(max_rows)
        truncated = True

    values = body.where(pd.notna(body), None).to_numpy(dtype=object)
    for row_offset in range(values.shape[0]):
        target_row = header_row + 1 + row_offset
        for col_offset in range(values.shape[1]):
            ws.cell(
                row=target_row, column=start_col + col_offset,
                value=_excel_safe(values[row_offset, col_offset]),
            )

    rows_written = int(values.shape[0])
    first_data_row = header_row + 1
    last_row = header_row + rows_written
    last_col = start_col + len(columns) - 1
    column_index = {name: start_col + offset for offset, name in enumerate(columns)}

    # number formats: per-cell up to the limit, else column-level
    if number_formats:
        heavy = rows_written > style_row_limit
        for name, fmt in number_formats.items():
            if name not in column_index:
                continue
            index = column_index[name]
            letter = get_column_letter(index)
            if heavy:
                ws.column_dimensions[letter].number_format = fmt
                continue
            for row in range(first_data_row, last_row + 1):
                ws.cell(row=row, column=index).number_format = fmt

    return {
        "header_row": header_row,
        "first_data_row": first_data_row,
        "last_row": last_row,
        "last_col": last_col,
        "columns": column_index,
        "rows_written": rows_written,
        "truncated": truncated,
        "source_rows": int(len(frame)),
    }


def _excel_safe(value: Any) -> Any:
    """Coerce a value into something openpyxl can write."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return None
        if isinstance(value, str) and len(value) > 32_000:
            return value[:31_997] + "..."
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        try:
            return _excel_safe(value.item())
        except Exception:  # pragma: no cover
            pass
    return str(value)


def column_ref(layout: Mapping[str, Any], name: str) -> Optional[str]:
    """``'C2:C500'`` range string for one column of a written block."""
    index = layout["columns"].get(name)
    if index is None or layout["rows_written"] == 0:
        return None
    letter = get_column_letter(index)
    return f"{letter}{layout['first_data_row']}:{letter}{layout['last_row']}"


def block_ref(layout: Mapping[str, Any], names: Sequence[str]) -> Optional[str]:
    """Range spanning several (contiguous or not) columns, by min/max index."""
    indices = [layout["columns"][n] for n in names if n in layout["columns"]]
    if not indices or layout["rows_written"] == 0:
        return None
    return (
        f"{get_column_letter(min(indices))}{layout['first_data_row']}:"
        f"{get_column_letter(max(indices))}{layout['last_row']}"
    )


# ---------------------------------------------------------------------------
# Conditional formatting
# ---------------------------------------------------------------------------


def apply_severity_rules(ws: Worksheet, ref: Optional[str]) -> None:
    """Colour GREEN/AMBER/RED/GREY text cells, keeping the label visible."""
    if not ref:
        return
    for label in ("RED", "AMBER", "GREEN", "GREY"):
        ws.conditional_formatting.add(
            ref,
            CellIsRule(
                operator="equal",
                formula=[f'"{label}"'],
                fill=severity_fill(label),
                font=severity_font(label),
            ),
        )


def apply_severity_row_rules(
    ws: Worksheet, ref: Optional[str], severity_column_letter: str, first_row: int
) -> None:
    """Tint a whole row according to its severity column."""
    if not ref:
        return
    for label in ("RED", "AMBER"):
        ws.conditional_formatting.add(
            ref,
            FormulaRule(
                formula=[f'${severity_column_letter}{first_row}="{label}"'],
                fill=severity_fill(label),
                stopIfTrue=False,
            ),
        )


def apply_threshold_rules(
    ws: Worksheet, ref: Optional[str], amber: float, red: float
) -> None:
    """Numeric 'higher is worse' banding, e.g. for PSI/CSI value columns."""
    if not ref:
        return
    ws.conditional_formatting.add(
        ref,
        CellIsRule(
            operator="greaterThanOrEqual", formula=[str(red)],
            fill=severity_fill("RED"), font=severity_font("RED"),
        ),
    )
    ws.conditional_formatting.add(
        ref,
        CellIsRule(
            operator="between", formula=[str(amber), str(red)],
            fill=severity_fill("AMBER"), font=severity_font("AMBER"),
        ),
    )
    ws.conditional_formatting.add(
        ref,
        CellIsRule(
            operator="lessThan", formula=[str(amber)],
            fill=severity_fill("GREEN"), font=severity_font("GREEN"),
        ),
    )


def apply_color_scale(ws: Worksheet, ref: Optional[str], *, reverse: bool = False) -> None:
    """Red-amber-green colour scale (``reverse=True`` for 'higher is better')."""
    if not ref:
        return
    low, high = ("63BE7B", "F8696B") if not reverse else ("F8696B", "63BE7B")
    ws.conditional_formatting.add(
        ref,
        ColorScaleRule(
            start_type="min", start_color=low,
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color=high,
        ),
    )


def apply_diverging_scale(ws: Worksheet, ref: Optional[str]) -> None:
    """Green-white-red scale centred on zero, for signed change columns."""
    if not ref:
        return
    ws.conditional_formatting.add(
        ref,
        ColorScaleRule(
            start_type="min", start_color="63BE7B",
            mid_type="num", mid_value=0, mid_color="FFFFFF",
            end_type="max", end_color="F8696B",
        ),
    )


def apply_data_bars(ws: Worksheet, ref: Optional[str], colour: str = "638EC6") -> None:
    if not ref:
        return
    ws.conditional_formatting.add(
        ref, DataBarRule(start_type="min", end_type="max", color=colour, showValue=True)
    )


def apply_heatmap(ws: Worksheet, ref: Optional[str]) -> None:
    """White-to-navy intensity scale, used for migration matrices."""
    if not ref:
        return
    ws.conditional_formatting.add(
        ref,
        ColorScaleRule(
            start_type="num", start_value=0, start_color="FFFFFF",
            mid_type="percentile", mid_value=60, mid_color="9DC3E6",
            end_type="max", end_color="1F3864",
        ),
    )


# ---------------------------------------------------------------------------
# Finishing
# ---------------------------------------------------------------------------


def autosize_columns(
    ws: Worksheet,
    frame: pd.DataFrame,
    *,
    start_col: int = 1,
    min_width: int = 9,
    max_width: int = 52,
    sample_rows: int = 400,
    overrides: Optional[Mapping[str, int]] = None,
) -> None:
    """Width from the header plus a sample of the data (bounded)."""
    overrides = overrides or {}
    sample = frame.head(sample_rows)
    for offset, name in enumerate([str(c) for c in frame.columns]):
        letter = get_column_letter(start_col + offset)
        if name in overrides:
            ws.column_dimensions[letter].width = overrides[name]
            continue
        header_width = min(26, max(len(part) for part in name.split("_")) + 4)
        try:
            body = sample[frame.columns[offset]].astype("string").dropna()
            content_width = int(body.str.len().quantile(0.92)) + 2 if not body.empty else 0
        except Exception:  # pragma: no cover
            content_width = 0
        ws.column_dimensions[letter].width = max(
            min_width, min(max_width, max(header_width, content_width))
        )


def finalise(
    ws: Worksheet,
    layout: Mapping[str, Any],
    *,
    freeze: bool = True,
    autofilter: bool = True,
    table_name: Optional[str] = None,
    table_row_limit: int = 50_000,
    freeze_columns: int = 0,
) -> None:
    """Freeze panes, add an autofilter and (optionally) a real Excel table."""
    if layout["rows_written"] == 0:
        return
    if freeze:
        ws.freeze_panes = ws.cell(
            row=layout["first_data_row"], column=freeze_columns + 1
        ).coordinate
    ref = (
        f"{get_column_letter(1)}{layout['header_row']}:"
        f"{get_column_letter(layout['last_col'])}{layout['last_row']}"
    )
    if table_name and layout["rows_written"] <= table_row_limit:
        try:
            table = Table(displayName=sanitise_table_name(table_name), ref=ref)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleLight8", showFirstColumn=False, showLastColumn=False,
                showRowStripes=True, showColumnStripes=False,
            )
            ws.add_table(table)
            return   # an Excel table carries its own filter dropdowns
        except Exception:  # pragma: no cover - duplicate names, odd refs
            pass
    if autofilter:
        ws.auto_filter.ref = ref


def write_truncation_banner(
    ws: Worksheet, row: int, layout: Mapping[str, Any], sidecar: Optional[str], *, width: int = 12
) -> int:
    """Explicit, unmissable note when a sheet could not hold every row."""
    if not layout.get("truncated"):
        return row
    message = (
        f"NOTE: this sheet shows the first {layout['rows_written']:,} of "
        f"{layout['source_rows']:,} rows (Excel row budget). No data has been dropped from the "
        "analysis"
        + (f"; the complete table was written to '{sidecar}'." if sidecar else
           "; raise excel.max_rows_per_sheet or narrow runtime.csi_variable_whitelist to see "
           "the rest in Excel.")
    )
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(2, width))
    cell = ws.cell(row=row, column=1, value=message)
    cell.font = Font(bold=True, size=10, color=SEVERITY_FONT["AMBER"])
    cell.fill = severity_fill("AMBER")
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    return row + 1


__all__ = [
    "NAVY", "BAND", "SEVERITY_FILL", "SEVERITY_FONT", "RISK_RAMP",
    "FORMAT_INDEX", "FORMAT_INDEX_SHORT", "FORMAT_PCT", "FORMAT_PCT_SIGNED",
    "FORMAT_PP_SIGNED", "FORMAT_COUNT", "FORMAT_PROPORTION", "FORMAT_RATIO", "FORMAT_DATE",
    "sanitise_sheet_name", "sanitise_table_name", "risk_fill_for_rank",
    "severity_fill", "severity_font", "write_title", "write_section", "write_note",
    "write_key_values", "write_frame", "column_ref", "block_ref",
    "apply_severity_rules", "apply_severity_row_rules", "apply_threshold_rules",
    "apply_color_scale", "apply_diverging_scale", "apply_data_bars", "apply_heatmap",
    "autosize_columns", "finalise", "write_truncation_banner",
]
