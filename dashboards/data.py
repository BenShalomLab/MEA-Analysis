"""Read-only checkpoint loaders for dashboard pages.

Mirrors the checkpoint parsing logic from
streamlit_checkpoint_analyzer/checkpoint_dashboard.py but returns plain
DataFrames with no Streamlit cache decorators.
"""

from __future__ import annotations

import json
import sys
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


def load_checkpoints(checkpoint_dir: str | Path) -> pd.DataFrame:
    """Scan checkpoint_dir for JSON files; return tidy DataFrame.

    Falls back to recursive search if no files found at top level.
    Never raises — returns empty DataFrame on missing/empty dir.
    """
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
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.DataFrame(rows)


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

    Handles both v1 (burstlets/count/rate/duration) and
    v2 (burst_fragments/burst_count/burst_rate_hz/burst_duration_s) schemas.
    """
    bl_sec = (raw.get("burst_fragments") or raw.get("burstlets") or {})
    bl   = bl_sec.get("metrics", {})
    nb   = (raw.get("network_bursts", {}) or {}).get("metrics", {})
    sb   = (raw.get("superbursts", {}) or {}).get("metrics", {})
    diag = raw.get("diagnostics", {}) or {}

    def _count(m):  return m.get("burst_count") or m.get("count") or 0
    def _rate(m):   return float(m.get("burst_rate_hz") or m.get("rate") or 0)
    def _dur(m):    return float(
        ((m.get("burst_duration_s") or m.get("duration") or {}).get("mean")) or 0
    )

    return {
        "n_units":              raw.get("n_units") or diag.get("n_units"),
        "n_bursty_units":       diag.get("n_bursty_units"),
        "burstlets_count":      _count(bl),
        "network_bursts_count": _count(nb),
        "superbursts_count":    _count(sb),
        "burst_rate_hz":        round(_rate(nb), 4),
        "mean_burst_dur_s":     round(_dur(nb), 3),
        "adaptive_bin_ms":      diag.get("bin_size_ms") or diag.get("adaptive_bin_ms"),
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
