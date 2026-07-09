"""Read-only checkpoint loaders for dashboard pages.

Mirrors the checkpoint parsing logic from
streamlit_checkpoint_analyzer/checkpoint_dashboard.py but returns plain
DataFrames with no Streamlit cache decorators.
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import pandas as pd

# Ensure repo root on path when run as a module from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mea_checkpoint import ProcessingStage  # noqa: E402

STAGE_MAP: dict[int, str] = {s.value: s.name for s in ProcessingStage}

LEGACY_STAGE_MAP: dict[int, str] = {
    0: "NOT_STARTED",
    1: "PREPROCESSING",
    2: "PREPROCESSING_COMPLETE",
    3: "SORTING",
    4: "SORTING_COMPLETE",
    5: "ANALYZER",
    6: "ANALYZER_COMPLETE",
    7: "REPORTS",
    8: "REPORTS_COMPLETE",
}

IN_PROGRESS_STAGES = frozenset({"PREPROCESSING", "SORTING", "MERGE", "ANALYZER", "REPORTS"})
COMPLETE_STAGES = frozenset({
    "PREPROCESSING_COMPLETE", "SORTING_COMPLETE", "MERGE_COMPLETE",
    "ANALYZER_COMPLETE", "REPORTS_COMPLETE",
})
TERMINAL_STAGE = "REPORTS_COMPLETE"

# (stage_name, threshold stage_num for "complete", running stage_num)
STAGE_COLS = [
    ("preproc",  2, 1),
    ("sorting",  4, 3),
    ("merge",    6, 5),
    ("analyzer", 8, 7),
    ("reports",  10, 9),
]

_EMPTY_COLS = [
    "file", "path", "project", "date", "chip", "run", "well", "rec",
    "stage", "stage_num", "failed", "error", "num_units",
    "analyzer_folder", "data_dir", "last_updated",
]


def _safe(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def load_checkpoints(checkpoint_dir: str | Path, *, force: bool = False) -> pd.DataFrame:
    """Scan checkpoint_dir for JSON files; return tidy DataFrame.

    Falls back to recursive search if no files found at top level.
    Never raises — returns empty DataFrame on missing/empty dir.
    Results cached for _CP_CACHE_TTL seconds; pass force=True to bypass.
    """
    _cp_key = str(checkpoint_dir)
    if not force:
        _cp_entry = _CP_CACHE.get(_cp_key)
        if _cp_entry and (_time.monotonic() - _cp_entry[0]) < _CP_CACHE_TTL:
            return _cp_entry[1]

    root = Path(checkpoint_dir)
    if not root.exists():
        return pd.DataFrame(columns=_EMPTY_COLS)

    files = sorted(root.glob("*.json"))
    if not files:
        files = sorted(root.rglob("*checkpoint*.json"))
    if not files:
        files = sorted(root.rglob("*.json"))

    rows = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue

        schema_v = 1
        try:
            schema_v = int(raw.get("checkpoint_schema_version", 1) or 1)
        except (ValueError, TypeError):
            pass
        smap = LEGACY_STAGE_MAP if schema_v < 2 else STAGE_MAP

        stage_num = raw.get("stage")
        try:
            stage_num = int(stage_num)
        except (ValueError, TypeError):
            stage_num = None

        stage_name = smap.get(stage_num, "UNKNOWN") if stage_num is not None else "UNKNOWN"

        failed_stage_raw = raw.get("failed_stage")
        failed = (
            bool(raw.get("failed", False))
            or failed_stage_raw is not None
            or raw.get("error") is not None
        )
        if failed_stage_raw is not None:
            try:
                failed_stage_raw = int(failed_stage_raw)
            except (ValueError, TypeError):
                pass
            stage_name = f"FAILED_AT_{smap.get(failed_stage_raw, failed_stage_raw)}"

        rows.append({
            "file": f.name,
            "path": str(f.resolve()),
            "project": _safe(raw, "project_name", "project"),
            "date": raw.get("date"),
            "chip": _safe(raw, "chip_id", "chip"),
            "run": raw.get("run_id"),
            "well": _safe(raw, "well_id", "well"),
            "rec": raw.get("rec_name"),
            "stage": stage_name,
            "stage_num": stage_num,
            "failed": failed,
            "error": raw.get("error"),
            "num_units": _safe(raw, "num_units_filtered", "num_units", "n_units"),
            "analyzer_folder": _safe(raw, "analyzer_folder", "output_dir"),
            "data_dir": raw.get("data_dir"),
            "last_updated": raw.get("last_updated"),
        })

    if not rows:
        _cp_df = pd.DataFrame(columns=_EMPTY_COLS)
    else:
        _cp_df = pd.DataFrame(rows)
    _CP_CACHE[_cp_key] = (_time.monotonic(), _cp_df)
    return _cp_df


def checkpoint_kpis(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"total": 0, "complete": 0, "running": 0, "failed": 0, "not_started": 0}
    return {
        "total": len(df),
        "complete": int((df["stage"] == TERMINAL_STAGE).sum()),
        "running": int(df["stage"].isin(IN_PROGRESS_STAGES).sum()),
        "failed": int(df["failed"].sum()),
        "not_started": int((df["stage"] == "NOT_STARTED").sum()),
    }


def stage_throughput(df: pd.DataFrame) -> list[dict]:
    """Per-stage complete/running/not_run counts for home throughput table."""
    total = len(df) if not df.empty else 0
    if total == 0:
        return [{"name": c, "complete": 0, "running": 0, "not_run": 0, "total": 0}
                for c, _, _ in STAGE_COLS]
    sn = df["stage_num"].fillna(-1).astype(int)
    failed = df["failed"].fillna(False).astype(bool)
    rows = []
    for col, complete_thresh, running_val in STAGE_COLS:
        complete = int((sn >= complete_thresh).sum())
        running = int(((sn == running_val) & ~failed).sum())
        rows.append({
            "name": col,
            "complete": complete,
            "running": running,
            "not_run": max(0, total - complete - running),
            "total": total,
        })
    return rows


def delete_checkpoint(path: str | Path) -> tuple[bool, str]:
    """Permanently delete a checkpoint JSON file. Returns (ok, error_message)."""
    path = Path(path)
    try:
        if not path.exists():
            return False, "File not found."
        path.unlink()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def bulk_delete_checkpoints(
    df: pd.DataFrame,
    filter_stage: str | None = None,
    filter_failed_only: bool = False,
) -> tuple[int, int]:
    """Delete matching checkpoint files. Returns (n_ok, n_fail)."""
    subset = df.copy()
    if filter_stage and filter_stage not in ("", "all"):
        subset = subset[subset["stage"] == filter_stage]
    if filter_failed_only:
        subset = subset[subset["failed"] == True]  # noqa: E712
    ok = fail = 0
    for path in subset["path"]:
        success, _ = delete_checkpoint(path)
        if success:
            ok += 1
        else:
            fail += 1
    return ok, fail


def reset_checkpoint(path: str | Path, to_stage_num: int) -> tuple[bool, str]:
    """Set checkpoint stage to to_stage_num and clear failure state."""
    import datetime as _dt
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
        raw["stage"] = to_stage_num
        raw.pop("failed", None)
        raw.pop("failed_stage", None)
        raw.pop("error", None)
        raw["last_updated"] = _dt.datetime.now().isoformat()
        path.write_text(json.dumps(raw, indent=2))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def bulk_reset_checkpoints(
    df: pd.DataFrame,
    to_stage_num: int,
    filter_stage: str | None = None,
    filter_failed_only: bool = False,
) -> tuple[int, int]:
    """Reset matching checkpoints. Returns (n_ok, n_fail)."""
    subset = df.copy()
    if filter_stage and filter_stage not in ("", "all"):
        subset = subset[subset["stage"] == filter_stage]
    if filter_failed_only:
        subset = subset[subset["failed"] == True]  # noqa: E712
    ok = fail = 0
    for path in subset["path"]:
        success, _ = reset_checkpoint(path, to_stage_num)
        if success:
            ok += 1
        else:
            fail += 1
    return ok, fail


def _parse_network_raw(raw: dict) -> dict:
    """Extract dashboard summary fields from a network_results.json dict.

    Uses the canonical schema produced by parameter_free_burst_detector.py.
    """
    bl   = (raw.get("burst_fragments") or {}).get("metrics") or {}
    nb   = (raw.get("network_bursts")  or {}).get("metrics") or {}
    sb   = (raw.get("superbursts")     or {}).get("metrics") or {}
    diag = raw.get("diagnostics") or {}

    return {
        "n_units":              raw.get("n_units") or diag.get("n_units"),
        "n_bursty_units":       diag.get("n_bursty_units"),
        "burstlets_count":      bl.get("burst_count", 0),
        "network_bursts_count": nb.get("burst_count", 0),
        "superbursts_count":    sb.get("burst_count", 0),
        "burst_rate_hz":        round(float(nb.get("burst_rate_hz") or 0), 4),
        "mean_burst_dur_s":     round(float((nb.get("burst_duration_s") or {}).get("mean") or 0), 3),
        "adaptive_bin_ms":      diag.get("bin_size_ms"),
        "_raw":                 raw,
    }


def load_network_results_from_checkpoints(df: pd.DataFrame) -> list[dict]:
    """Load network_results.json using output paths already stored in checkpoints."""
    rows = []
    for r in df.itertuples(index=False):
        folder = getattr(r, "analyzer_folder", None)
        if not folder:
            continue
        f = Path(folder) / "network_results.json"
        if not f.exists():
            continue
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue

        parsed = _parse_network_raw(raw)
        rows.append({
            "path":    str(folder),
            "project": r.project,
            "date":    r.date,
            "chip":    r.chip,
            "run":     r.run,
            "well":    r.well,
            **parsed,
        })
    return rows


def load_network_results(output_root: str | Path) -> list[dict]:
    """Walk output_root for network_results.json; return per-well rows."""
    root = Path(output_root)
    if not root.exists():
        return []
    rows = []
    for f in sorted(root.rglob("network_results.json")):
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue
        parts = f.parent.parts
        well    = parts[-1] if len(parts) >= 1 else "?"
        run     = parts[-2] if len(parts) >= 2 else "?"
        chip    = parts[-3] if len(parts) >= 3 else "?"
        date    = parts[-4] if len(parts) >= 4 else "?"
        project = parts[-5] if len(parts) >= 5 else "?"

        parsed = _parse_network_raw(raw)
        rows.append({
            "path":    str(f.parent),
            "project": project,
            "date":    date,
            "chip":    chip,
            "run":     run,
            "well":    well,
            **parsed,
        })
    return rows


# ── Network-results cache ─────────────────────────────────────────────────────
# Avoids re-reading hundreds of JSONs on every interval tick.
# TTL matches the dashboard auto-refresh interval (60 s).
# Manual Refresh buttons call invalidate_network_cache() to force a reload.

_NET_CACHE: dict[str, tuple[float, list]] = {}
_NET_CACHE_TTL = 55.0  # seconds

_CP_CACHE: dict[str, tuple[float, "pd.DataFrame"]] = {}
_CP_CACHE_TTL = 55.0  # seconds


def invalidate_network_cache() -> None:
    _NET_CACHE.clear()


def invalidate_checkpoint_cache() -> None:
    _CP_CACHE.clear()


def load_network_rows(
    checkpoint_dir_or_root: str | Path,
    *,
    from_checkpoints: bool = True,
    force: bool = False,
) -> list[dict]:
    """Cached call to load_network_results_from_checkpoints or load_network_results.

    Returns cached rows when called within _NET_CACHE_TTL seconds of the last
    read.  Pass force=True to bypass the TTL (e.g. on a manual Refresh click).
    """
    key = str(checkpoint_dir_or_root)
    if not force:
        entry = _NET_CACHE.get(key)
        if entry and (_time.monotonic() - entry[0]) < _NET_CACHE_TTL:
            return entry[1]

    if from_checkpoints:
        rows = load_network_results_from_checkpoints(load_checkpoints(key))
    else:
        rows = load_network_results(key)

    _NET_CACHE[key] = (_time.monotonic(), rows)
    return rows


def stage_cell_status(stage_num: int | None, failed: bool, failed_stage_num: int | None) -> dict[str, str]:
    """Return {col_name: css_class} for the 5 pipeline stage columns."""
    out: dict[str, str] = {}
    for col, complete_thresh, running_val in STAGE_COLS:
        if failed and failed_stage_num is not None:
            # Determine which stage failed
            for c2, ct2, rv2 in STAGE_COLS:
                if failed_stage_num == rv2:
                    out[col] = "failed" if c2 == col else (
                        "complete" if complete_thresh <= ct2 else "not_run"
                    )
                    break
            else:
                out[col] = "not_run"
        elif stage_num is None:
            out[col] = "not_run"
        elif stage_num >= complete_thresh:
            out[col] = "complete"
        elif stage_num == running_val:
            out[col] = "running"
        else:
            out[col] = "not_run"
    return out
