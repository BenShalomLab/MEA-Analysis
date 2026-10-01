#!/usr/bin/env python3
"""Collect network_results.json files and export per-project CSVs.

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
                  coactive_fraction_peak (alias of coactive_fraction_peak)
  - diagnostics keys: bin_size_ms, reference_isi_s, coactive_fraction_baseline,
                      detection_threshold, fragment_merge_gap_s, nb_merge_gap_s,
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
    "fragment_merge_gap_s", "fragment_merge_gap_source",
    "nb_merge_gap_s", "nb_merge_gap_source",
    "superburst_min_dur_s", "superburst_min_components", "superburst_merge_gap_s",
    "duration_source", "recording_duration_s", "analysis_window_s",
    "sigma_coactivity_bins", "sigma_firing_rate_bins",
]

# Event fields for which to compute percentile distributions
_EVT_DISTRIBUTION_FIELDS = {
    "burst_duration_s",
    "participation_fraction",
    "spikes_per_burst",
    "spikes_per_burst_per_unit",
    "intraburst_rate_hz",
    "burst_peak_hz_per_unit",
    "coactive_fraction_peak",
    "coactive_fraction_max",
    "burst_area_spikes_per_unit",
    "rise_time_s",
    "decay_time_s",
    "intraburst_isi_mean_s",
}


# ── Section helpers ───────────────────────────────────────────────────────────

def _flatten_section_metrics(metrics: dict, prefix: str) -> dict:
    """Flatten a section's metrics dict into prefixed columns.

    Scalar values → single column.
    Stats-dict values (mean/std/cv) → three columns.
    """
    row: dict = {}
    for key, val in metrics.items():
        col = f"{prefix}_{key}"
        if isinstance(val, dict):
            row[f"{col}_mean"] = val.get("mean")
            row[f"{col}_std"]  = val.get("std")
            row[f"{col}_cv"]   = val.get("cv")
        else:
            row[col] = val
    return row


def _flatten_events_distributions(events: list[dict], prefix: str) -> dict:
    """Compute per-event percentile distributions for key fields.

    For each field in _EVT_DISTRIBUTION_FIELDS, outputs:
      {prefix}_{field}_n, _min, _p25, _p50, _p75, _p95, _max
    """
    if not events:
        return {}
    row: dict = {}
    present = {k for ev in events for k, v in ev.items()
               if isinstance(v, (int, float)) and k in _EVT_DISTRIBUTION_FIELDS}
    for field in sorted(present):
        vals = [ev[field] for ev in events if isinstance(ev.get(field), (int, float))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        col = f"{prefix}_{field}"
        row[f"{col}_n"]   = len(arr)
        row[f"{col}_min"] = float(np.min(arr))
        row[f"{col}_p25"] = float(np.percentile(arr, 25))
        row[f"{col}_p50"] = float(np.percentile(arr, 50))
        row[f"{col}_p75"] = float(np.percentile(arr, 75))
        row[f"{col}_p95"] = float(np.percentile(arr, 95))
        row[f"{col}_max"] = float(np.max(arr))
    return row


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
    row: dict = {"output_dir": str(well_dir)}
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
        sec     = raw.get(section_key) or {}
        metrics = sec.get("metrics") or {}
        events  = sec.get("events") or []
        row.update(_flatten_section_metrics(metrics, prefix))
        row.update(_flatten_events_distributions(events, prefix))

    diag = raw.get("diagnostics") or {}
    for k in _DIAG_KEYS:
        if k != "n_units" and k in diag:
            row[f"diag_{k}"] = diag[k]

    return row


# ── Collection ────────────────────────────────────────────────────────────────

def collect(root: Path) -> list[dict]:
    return [extract_row(f) for f in sorted(root.rglob("network_results.json"))]


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
        nf = Path(out_dir) / "network_results.json"
        if nf.exists():
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

def to_dataframes(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """Return {project_name: DataFrame}, plus an "ALL" key for the combined table."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    id_cols = ["project", "date", "chip", "run", "well", "n_units"]
    if "data_dir" in df.columns:
        id_cols.append("data_dir")
    id_cols.append("output_dir")
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
               combined: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if combined:
        p = out_dir / "network_metrics_all.csv"
        dfs["ALL"].to_csv(p, index=False)
        written.append(p)
    else:
        for name, df in dfs.items():
            if name == "ALL":
                continue
            safe = name.replace("/", "_").replace(" ", "_")
            p = out_dir / f"network_metrics_{safe}.csv"
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
    args = parser.parse_args()

    if args.root:
        rows = collect(Path(args.root))
    else:
        rows = collect_from_checkpoints(Path(args.checkpoint_dir))

    if not rows:
        print("No network_results.json files found.")
        return 1

    dfs = to_dataframes(rows)
    written = write_csvs(dfs, Path(args.out_dir), combined=args.combined)

    total = len(dfs.get("ALL", pd.DataFrame()))
    print(f"Collected {total} wells across {len(dfs) - 1} project(s).")
    for p in written:
        df = pd.read_csv(p)
        print(f"  {p}  ({len(df)} rows × {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
