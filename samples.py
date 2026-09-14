# ==========================================================
# samples.py
#
# Experimental metadata lookup: samples.csv -> per-well sample record.
#
# The pipeline infers project/date/chip/run/well from the directory layout,
# which says nothing about what was actually cultured in each well. Genotype,
# cell line, prep type, batch and plating date cannot be derived from the
# recording, and without them no group comparison is possible: the analysis
# ends at "here are the metrics for well000 of chip 16657".
#
# This module reads one table describing the experiment and attaches the
# matching row to each well's results. It never raises: a missing or
# unreadable table degrades to an unmatched record and a warning, because
# losing metadata should not abort a multi-day batch run.
#
# See docs/samples_schema.md for the column contract.
# ==========================================================

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

# Columns used to decide which row describes a given recording. A row only
# needs the ones that matter for it: a row specifying just `chip` applies to
# every well of that chip, while one specifying `chip` and `well` applies to
# that well alone and wins over the broader row.
KEY_COLUMNS = ("project", "date", "chip", "run", "well")

# Recognised metadata columns. Anything else in the file is carried through
# unchanged, so the lab can add columns without editing this module.
KNOWN_METADATA_COLUMNS = (
    "line",
    "genotype",
    "prep_type",
    "batch",
    "plating_date",
    "div",
    "density_cells_per_mm2",
    "media",
    "treatment",
    "treatment_concentration",
    "notes",
)

# Ordered by descending specificity of the literal text, so that an
# unambiguous 4-digit year is tried before a 2-digit one.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y_%m_%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%y-%m-%d",
    "%y%m%d",
)


def parse_date(value):
    """Parse a date written in any of the formats the lab uses. None if not a date.

    Accepts datetime/date objects unchanged so callers can pass values that
    pandas has already converted.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    # A bare digit run is ambiguous between YYMMDD and YYYYMMDD; its length
    # decides, so try the matching format first rather than in list order.
    if text.isdigit():
        ordered = ("%Y%m%d", "%y%m%d") if len(text) == 8 else ("%y%m%d", "%Y%m%d")
        for fmt in ordered:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Last resort: a date embedded in a longer string, e.g. "2025-01-30_run3".
    match = re.search(r"(\d{4})[-_/.](\d{1,2})[-_/.](\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def compute_div(plating_date, recording_date):
    """Days in vitro between plating and recording. None if either is unknown.

    Negative values are returned as-is rather than suppressed: a recording
    dated before its own plating date means the table or the directory name
    is wrong, and silently hiding that would let it reach the statistics.
    """
    plated = parse_date(plating_date)
    recorded = parse_date(recording_date)
    if plated is None or recorded is None:
        return None
    return (recorded - plated).days


def _norm(value):
    if value is None:
        return ""
    return str(value).strip().casefold()


def _norm_well(value):
    """well000, well0, 000 and 0 all name the same well."""
    text = _norm(value)
    if not text:
        return ""
    text = text.removeprefix("well")
    stripped = text.lstrip("0")
    return stripped if stripped else "0"


def _norm_key(column, value):
    if column == "well":
        return _norm_well(value)
    if column == "date":
        parsed = parse_date(value)
        return parsed.isoformat() if parsed else _norm(value)
    text = _norm(value)
    # Chip and run ids are numeric but written with varying zero padding.
    if column in ("chip", "run") and text.isdigit():
        return str(int(text))
    return text


def load_samples(path):
    """Read the sample table. Returns (rows, error_message).

    Supports CSV/TSV directly and .xlsx through pandas when it is installed.
    """
    if path is None:
        return [], None

    table_path = Path(path)
    if not table_path.exists():
        return [], f"sample table not found: {table_path}"

    try:
        if table_path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            import pandas as pd

            frame = pd.read_excel(table_path, dtype=str)
            rows = frame.where(frame.notna(), None).to_dict(orient="records")
        else:
            delimiter = "\t" if table_path.suffix.lower() in (".tsv", ".tab") else ","
            with open(table_path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))
    except Exception as e:
        return [], f"could not read sample table {table_path}: {e}"

    cleaned = []
    for row in rows:
        if not row:
            continue
        # Normalise header spelling once, here, so the rest of the module can
        # assume lowercase underscore names.
        item = {}
        for key, value in row.items():
            if key is None:
                continue
            name = str(key).strip().casefold().replace(" ", "_")
            if name.startswith("_"):
                continue
            item[name] = None if value is None else str(value).strip()
        if any(v for v in item.values()):
            cleaned.append(item)

    return cleaned, None


def lookup_sample(rows, *, project=None, date=None, chip=None, run=None, well=None):
    """Most specific matching row, or None.

    A row matches when every key column it fills in equals the corresponding
    recording value; blank key cells are wildcards. The row that pins the most
    key columns wins, so broad defaults and per-well exceptions can coexist in
    one file. Returns (row, matched_columns, n_candidates).
    """
    recording = {
        "project": project,
        "date": date,
        "chip": chip,
        "run": run,
        "well": well,
    }
    normalised_recording = {c: _norm_key(c, recording[c]) for c in KEY_COLUMNS}

    candidates = []
    for row in rows:
        matched_columns = []
        for column in KEY_COLUMNS:
            cell = row.get(column)
            if cell is None or not str(cell).strip():
                continue  # wildcard
            if _norm_key(column, cell) != normalised_recording[column]:
                break
            matched_columns.append(column)
        else:
            candidates.append((len(matched_columns), matched_columns, row))

    if not candidates:
        return None, [], 0

    best_specificity = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_specificity]
    return best[0][2], best[0][1], len(best)


def resolve_sample(samples_file, *, project=None, date=None, chip=None,
                   run=None, well=None, logger=None):
    """Build the `sample` block written into each well's results.

    Always returns a dict. `matched` says whether a row was found, so a
    downstream analysis can tell "wild type" from "nobody recorded what this
    well was".
    """
    def _log(level, message, *args):
        if logger is not None:
            getattr(logger, level)(message, *args)

    record = {
        "matched": False,
        "samples_file": str(samples_file) if samples_file else None,
        "matched_on": [],
        "div": None,
        "div_source": None,
    }

    if not samples_file:
        return record

    rows, error = load_samples(samples_file)
    if error:
        _log("warning", "Sample metadata unavailable (%s); continuing without it.", error)
        record["error"] = error
        return record

    row, matched_columns, n_equally_specific = lookup_sample(
        rows, project=project, date=date, chip=chip, run=run, well=well
    )

    if row is None:
        _log(
            "warning",
            "No row in %s matches project=%s date=%s chip=%s run=%s well=%s; "
            "this well will have no genotype or DIV.",
            samples_file, project, date, chip, run, well,
        )
        return record

    if n_equally_specific > 1:
        _log(
            "warning",
            "%d rows in %s match chip=%s well=%s equally specifically; using the first. "
            "Add a distinguishing key column to disambiguate.",
            n_equally_specific, samples_file, chip, well,
        )
        record["ambiguous_match"] = n_equally_specific

    record["matched"] = True
    record["matched_on"] = matched_columns
    for key, value in row.items():
        if key in KEY_COLUMNS:
            continue
        record[key] = value if (value is None or str(value).strip()) else None

    # An explicit div column wins: some experiments count from a differentiation
    # or thaw date that is not the plating date in this table.
    explicit_div = row.get("div")
    if explicit_div not in (None, ""):
        try:
            record["div"] = int(float(str(explicit_div).strip()))
            record["div_source"] = "column"
        except (TypeError, ValueError):
            _log("warning", "Could not read div=%r as a number; deriving it instead.", explicit_div)

    if record["div"] is None:
        derived = compute_div(row.get("plating_date"), date)
        if derived is not None:
            record["div"] = derived
            record["div_source"] = "plating_date"
            if derived < 0:
                _log(
                    "warning",
                    "DIV is negative (%d): recording date %s precedes plating date %s.",
                    derived, date, row.get("plating_date"),
                )

    _log(
        "info",
        "Sample: genotype=%s line=%s prep=%s batch=%s DIV=%s (matched on %s)",
        record.get("genotype"), record.get("line"), record.get("prep_type"),
        record.get("batch"), record.get("div"), matched_columns or "wildcard row",
    )
    return record
