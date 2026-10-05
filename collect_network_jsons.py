#!/usr/bin/env python3
"""Collect network_results.json files (and network_results_<detector>.json when
both burst detectors ran) and export per-project CSVs, one row per result file.

The bf_/nb_/sb_ columns hold each per-event field as a whole list in one cell
(JSON, e.g. "[0.41, 0.77]"; parse with json.loads). Statistics of those lists (mean/std/cv,
percentiles, duration max) are not written; the per-well scalars (burst_count, rates,
duty cycle), inter-burst-interval stats, diagnostics and spike participation are.

Uses the canonical schema produced by parameter_free_burst_detector.py:
  - burst_fragments / network_bursts / superbursts
  - metrics keys: burst_count, burst_rate_hz, burst_duration_s,
                  burst_duration_p95_s / _max_s,
                  ifbi_s / ibi_s / isbi_s and their _gap_s counterparts,
                  burst_area_spikes_per_unit, participation_fraction, spikes_per_burst,
                  spikes_per_burst_per_unit, intraburst_rate_hz,
                  burst_peak_hz_per_unit (per unit),
                  burst_peak_hz_array (array-wide),
                  coactive_fraction_peak, coactive_fraction_max,
  - diagnostics keys: bin_size_ms, reference_isi_s, coactive_fraction_baseline,
                      detection_threshold, merge_floor, superburst_gap_s,
                      coactive_fraction_bimodality, threshold_source, min_units_for_burst, …
  - n_units at top level

Usage
-----
# From the output root (AnalyzedData/…)
python collect_network_jsons.py --root /path/to/AnalyzedData --out-dir ./metrics

# Or point at the checkpoint dir so paths come from checkpoint JSON metadata
python collect_network_jsons.py --checkpoint-dir /path/to/checkpoints --out-dir ./metrics

# Single combined CSV instead of per-project files
python collect_network_jsons.py --root /path/to/AnalyzedData --out-dir ./metrics --combined

# One burst detector only (writes network_metrics_<project>_<detector>.csv)
python collect_network_jsons.py --root /path/to/AnalyzedData --out-dir ./metrics --detector parameter_free
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Section layout ────────────────────────────────────────────────────────────
# prefix → (section key in JSON, IBI metric key in that section)
_SECTIONS = {
    "bf": ("burst_fragments", "ifbi_s"),
    "nb": ("network_bursts",  "ibi_s"),
    "sb": ("superbursts",     "isbi_s"),
}

# Diagnostics to extract (flat scalars / strings)
_DIAG_KEYS = [
    "n_units", "n_bursty_units_by_isi_statistics",
    "bin_size_ms",
    "reference_isi_s", "reference_isi_source",
    "coactive_fraction_baseline", "coactive_fraction_mad", "coactive_fraction_bimodality",
    "burst_detection_valid",
    "detection_threshold", "threshold_source",
    "min_coactive_fraction", "min_units_for_burst",
    "fragment_merge_rule", "merge_floor", "fragment_max_gap_s",
    "nb_gap_bimodality", "superburst_gap_s", "superburst_gap_source", "elongated_min_dur_s",
    "superburst_min_dur_s", "superburst_min_components",
    "duration_source", "recording_duration_s", "analysis_window_s",
    "sigma_coactivity_bins", "sigma_firing_rate_bins",
]

# Scalar metrics that only summarise the per-event burst_duration_s list.
_DURATION_SUMMARY_STATS = {"burst_duration_p95_s", "burst_duration_max_s"}


# ── Section helpers ───────────────────────────────────────────────────────────

def _flatten_section_metrics(metrics: dict, events: list[dict], prefix: str) -> dict:
    """The section's own per-well metrics, minus statistics of the event lists.

    Kept: scalars (count, rates, duty cycle) and stats of values that are not an
    event field (inter-burst intervals). Dropped: mean/std/cv of a field that is
    written as a list, and the duration p95/max that summarise burst_duration_s.
    """
    listed = {k for ev in events for k, v in ev.items() if not isinstance(v, (dict, list))}
    row: dict = {}
    for key, val in metrics.items():
        if key in _DURATION_SUMMARY_STATS or (isinstance(val, dict) and key in listed):
            continue
        if isinstance(val, dict):
            for stat in ("mean", "std", "cv"):
                row[f"{prefix}_{key}_{stat}"] = val.get(stat)
        else:
            row[f"{prefix}_{key}"] = val
    return row


def _flatten_event_lists(events: list[dict], prefix: str) -> dict:
    """One column per per-event field, holding that field's values for every event
    of the well as a JSON list, e.g. nb_burst_duration_s = "[0.41, 0.77, 1.02]".

    Position i in every list is the same event. Counts, rates, means, percentiles and
    the like are not written here; compute them from the lists. A well with no events
    has no such columns.
    """
    fields: dict[str, None] = {}
    for ev in events:
        for k, v in ev.items():
            if not isinstance(v, (dict, list)):
                fields.setdefault(k)
    return {f"{prefix}_{field}": json.dumps([ev.get(field) for ev in events])
            for field in fields}


# ── Path metadata ─────────────────────────────────────────────────────────────

def _parse_path_metadata(well_dir: Path) -> dict:
    """Infer project/date/chip/run/well from the output directory path.

    Fallback only: extract_row prefers the ids recorded inside the JSON, which
    come from the recording's own metadata rather than from directory names.

    The layout mirrors the input tree, which ends in an assay folder:
      <output_root>/<project>/<date>/<chip>/<run>/Network/well000/
    so the run id is two levels above the well, not one. Reading it one level
    up yielded the literal string "Network" for every row.
    """
    parts = well_dir.parts
    return {
        "project": parts[-6] if len(parts) >= 6 else None,
        "date":    parts[-5] if len(parts) >= 5 else None,
        "chip":    parts[-4] if len(parts) >= 4 else None,
        "run":     parts[-3] if len(parts) >= 3 else None,
        "well":    parts[-1],
    }


# Ids written into the JSON by the pipeline, and the column each maps to.
_JSON_ID_KEYS = {
    "project": "project",
    "date":    "date",
    "chip":    "chip_id",
    "run":     "run_id",
    "well":    "well",
}


def _flatten_spike_participation(block: dict | None) -> dict:
    """Share of spiking inside network bursts -> sp_* columns.

    sp_percent_random_spikes is PRS in Mossink et al. 2021: a culture can
    hold its burst rate while its neurons drift out of the bursts, and only
    this ratio shows that.
    """
    if not isinstance(block, dict):
        return {}
    return {f"sp_{key}": value for key, value in block.items()}


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_row(json_path: Path) -> dict:
    """Flatten a single network_results.json into a row dict."""
    well_dir = json_path.parent
    row: dict = {"output_dir": str(well_dir), "result_file": json_path.name}
    row.update(_parse_path_metadata(well_dir))

    try:
        raw = json.loads(json_path.read_text())
    except Exception as exc:
        row["error"] = str(exc)
        return row

    # Ids recorded by the pipeline win over the ones guessed from the path.
    for column, json_key in _JSON_ID_KEYS.items():
        value = raw.get(json_key)
        if value not in (None, ""):
            row[column] = value

    row.update(_flatten_spike_participation(raw.get("spike_participation")))

    row["n_units"] = raw.get("n_units")
    row["schema_version"] = ((raw.get("diagnostics") or {}).get("schema_version"))
    row["detector"] = raw.get("detector") or ((raw.get("diagnostics") or {}).get("detector"))

    for prefix, (section_key, _ibi_key) in _SECTIONS.items():
        sec = raw.get(section_key) or {}
        events = sec.get("events") or []
        row.update(_flatten_section_metrics(sec.get("metrics") or {}, events, prefix))
        row.update(_flatten_event_lists(events, prefix))

    diag = raw.get("diagnostics") or {}
    for k in _DIAG_KEYS:
        if k != "n_units" and k in diag:
            row[f"diag_{k}"] = diag[k]

    return row


# ── Collection ────────────────────────────────────────────────────────────────

def _is_result_file(path: Path) -> bool:
    """network_results.json, or network_results_<detector>.json when several
    detectors ran. A leftover .tmp.json from an interrupted write is not a result."""
    return path.suffix == ".json" and not path.name.endswith(".tmp.json")


def collect(root: Path) -> list[dict]:
    files = [f for f in root.rglob("network_results*.json") if _is_result_file(f)]
    return [extract_row(f) for f in sorted(files)]


def collect_from_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """Use output_dir from checkpoint JSONs to locate network_results.json files."""
    rows = []
    for cp_file in sorted(checkpoint_dir.rglob("*.json")):
        try:
            cp = json.loads(cp_file.read_text())
        except Exception:
            continue
        out_dir = cp.get("output_dir") or cp.get("analyzer_folder")
        if not out_dir:
            continue
        for nf in sorted(Path(out_dir).glob("network_results*.json")):
            if not _is_result_file(nf):
                continue
            row = extract_row(nf)
            for key in ("project", "date", "chip", "run", "well"):
                cp_val = (cp.get(key) or cp.get(f"{key}_id")
                          or (cp.get("chip_id") if key == "chip" else None))
                if cp_val:
                    row[key] = cp_val
            if cp.get("data_dir"):
                row["data_dir"] = cp["data_dir"]
            rows.append(row)
    return rows


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def to_dataframes(rows: list[dict], detector: str | None = None) -> dict[str, pd.DataFrame]:
    """Return {project_name: DataFrame}, plus an "ALL" key for the combined table.

    With ``detector`` set, keep only rows from that burst detector (rows with no
    detector, i.e. pre-schema-5 files, are dropped) and drop columns that are
    entirely empty for it, so the table has no leftover columns of the other detector.
    """
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if detector is not None:
        if "detector" not in df.columns:
            return {}
        df = df[df["detector"] == detector]
        if df.empty:
            return {}
        df = df.dropna(axis=1, how="all").reset_index(drop=True)

    id_cols = ["project", "date", "chip", "run", "well", "n_units"]
    if "data_dir" in df.columns:
        id_cols.append("data_dir")
    id_cols.append("output_dir")
    if "result_file" in df.columns:
        id_cols.append("result_file")
    metric_cols = [c for c in df.columns if c not in id_cols and c != "error"]

    def _col_sort_key(c: str) -> tuple:
        if c.startswith("bf_"):   return (0, c)
        if c.startswith("nb_"):   return (1, c)
        if c.startswith("sb_"):   return (2, c)
        if c.startswith("diag_"): return (3, c)
        return (4, c)

    metric_cols = sorted(metric_cols, key=_col_sort_key)
    ordered = [c for c in id_cols if c in df.columns] + metric_cols
    if "error" in df.columns:
        ordered.append("error")
    df = df[ordered]

    result: dict[str, pd.DataFrame] = {"ALL": df}
    for proj, grp in df.groupby("project", dropna=False):
        result[str(proj or "unknown")] = grp.reset_index(drop=True)
    return result


def write_csvs(dfs: dict[str, pd.DataFrame], out_dir: Path,
               combined: bool = False, detector: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    suffix = f"_{detector}" if detector else ""
    if combined:
        p = out_dir / f"network_metrics_all{suffix}.csv"
        dfs["ALL"].to_csv(p, index=False)
        written.append(p)
    else:
        for name, df in dfs.items():
            if name == "ALL":
                continue
            safe = name.replace("/", "_").replace(" ", "_")
            p = out_dir / f"network_metrics_{safe}{suffix}.csv"
            df.to_csv(p, index=False)
            written.append(p)
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Output root to walk for network_results.json")
    src.add_argument("--checkpoint-dir",
                     help="Checkpoint directory (uses output_dir from each JSON)")
    parser.add_argument("--out-dir", default="./metrics",
                        help="Directory to write CSVs (default: ./metrics)")
    parser.add_argument("--combined", action="store_true",
                        help="Write a single combined CSV instead of one per project")
    parser.add_argument("--detector", choices=("parameter_free", "gaussian"),
                        help="Collect only this burst detector's results (default: every "
                             "result file, one row per detector per well). Rows without a "
                             "detector and columns empty for it are dropped.")
    args = parser.parse_args()

    if args.root:
        rows = collect(Path(args.root))
    else:
        rows = collect_from_checkpoints(Path(args.checkpoint_dir))

    if not rows:
        print("No network_results.json files found.")
        return 1

    dfs = to_dataframes(rows, detector=args.detector)
    if not dfs:
        print(f"No results for detector {args.detector!r}.")
        return 1
    written = write_csvs(dfs, Path(args.out_dir), combined=args.combined, detector=args.detector)

    total = len(dfs.get("ALL", pd.DataFrame()))
    print(f"Collected {total} wells across {len(dfs) - 1} project(s).")
    for p in written:
        df = pd.read_csv(p)
        print(f"  {p}  ({len(df)} rows × {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
