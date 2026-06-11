
#!/usr/bin/env python3
"""Collect MEA network JSON metrics into a CSV.

This script:
- recursively finds *network*.json files under a root folder
- extracts path metadata (Project, Date, Chip_ID, RunID, Well)
- parses network_bursts / superbursts / burstlets metrics
- writes one CSV row per JSON file

Usage:
    python collect_network_metrics.py \
        --root /path/to/AnalyzedData/ProjectFolder \
        --output metrics.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/mnt/Vol20tb1/user_workspaces/shruti/MEA_Analysis/MEA_Analysis_V2/MEA_Analysis/AnalyzedData/KCNT1_T4_C1_04122024/"
DEFAULT_OUTPUT = "mea_network_metrics.csv"
ANCHOR = "AnalyzedData"
WELL_RE = re.compile(r"^well\d+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MEA network JSON files and export extracted metrics to CSV."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Root directory to search recursively (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pattern",
        default="*network*.json",
        help="Glob pattern used for JSON discovery (default: *network*.json)",
    )
    parser.add_argument(
        "--anchor",
        default=ANCHOR,
        help="Optional path segment used to locate Project/Date/Chip_ID/RunID (default: AnalyzedData). Set to empty to disable anchor-based parsing.",
    )
    parser.add_argument(
        "--include-lists",
        action="store_true",
        help="Keep list-valued metrics as JSON strings in the CSV. If omitted, they are still serialized as JSON strings.",
    )
    return parser.parse_args()


def find_json_files(root: str, pattern: str) -> List[str]:
    search_path = os.path.join(root, "**", pattern)
    return sorted(glob.glob(search_path, recursive=True))


def parse_path_metadata(path: str, root: Optional[str] = None, anchor: Optional[str] = ANCHOR) -> Dict[str, Optional[str]]:
    parts = Path(path).parts
    out = {"Project": None, "Date": None, "Chip_ID": None, "RunID": None, "Well": None}

    candidates = []
    if root:
        try:
            candidates.append(Path(path).relative_to(root).parts)
        except ValueError:
            pass
    candidates.append(parts)

    for candidate in candidates:
        if anchor and anchor in candidate:
            idx = candidate.index(anchor)
            # Expected structure:
            # .../AnalyzedData/Project/Date/ChipID/Network/RunID/Well/...
            try:
                out["Project"] = candidate[idx + 1]
                out["Date"] = candidate[idx + 2]
                out["Chip_ID"] = candidate[idx + 3]
                out["RunID"] = candidate[idx + 5]
                out["Well"] = candidate[idx + 6]
                return out
            except IndexError:
                pass

        well_idx = next((i for i, part in enumerate(candidate) if WELL_RE.match(part)), None)
        if well_idx is not None and well_idx >= 4:
            out["Project"] = candidate[well_idx - 4]
            out["Date"] = candidate[well_idx - 3]
            out["Chip_ID"] = candidate[well_idx - 2]
            out["RunID"] = candidate[well_idx - 1]
            out["Well"] = candidate[well_idx]
            return out

    # Fallback: use nearby path parts if the anchor is missing
    if len(parts) >= 5:
        out["Project"] = parts[-5]
        out["Date"] = parts[-4]
        out["Chip_ID"] = parts[-3]
        out["RunID"] = parts[-2]
        out["Well"] = parts[-1]
    return out


def safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def serialize_list(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(list(value))
    return value


def extract_block_metrics(block: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        f"{prefix}_count": np.nan,
        f"{prefix}_rate_hz": np.nan,
        f"{prefix}_duration_mean_s": np.nan,
        f"{prefix}_ibi_mean_s": np.nan,
        f"{prefix}_spikes_per_burst_mean": np.nan,
        f"{prefix}_participation_mean": np.nan,
        f"{prefix}_burst_peak_mean": np.nan,
        f"{prefix}_peak_synchrony_mean": np.nan,
        f"{prefix}_fragment_count_mean": np.nan,
        f"{prefix}_durations_list": json.dumps([]),
        f"{prefix}_peak_list": json.dumps([]),
        f"{prefix}_intensity_list": json.dumps([]),
        f"{prefix}_synchrony_energy_list": json.dumps([]),
        f"{prefix}_fragment_count_list": json.dumps([]),
    }

    if not block:
        return metrics

    m = block.get("metrics", {}) if isinstance(block, dict) else {}
    events = block.get("events", []) if isinstance(block, dict) else []

    metrics[f"{prefix}_count"] = m.get("count", np.nan)
    if "rate" in m:
        metrics[f"{prefix}_rate_hz"] = m.get("rate", np.nan)
    metrics[f"{prefix}_duration_mean_s"] = safe_get(m, "duration", "mean")
    metrics[f"{prefix}_ibi_mean_s"] = safe_get(m, "inter_event_interval", "mean")
    metrics[f"{prefix}_spikes_per_burst_mean"] = safe_get(m, "spikes_per_burst", "mean")
    metrics[f"{prefix}_participation_mean"] = safe_get(m, "participation", "mean")
    metrics[f"{prefix}_burst_peak_mean"] = safe_get(m, "burst_peak", "mean")
    metrics[f"{prefix}_peak_synchrony_mean"] = safe_get(m, "peak_synchrony", "mean")

    durations: List[Any] = []
    peaks: List[Any] = []
    intensity: List[Any] = []
    synchrony_energy: List[Any] = []
    fragment_counts: List[Any] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if "duration_s" in ev:
            durations.append(ev["duration_s"])
        if "peak_synchrony" in ev:
            peaks.append(ev["peak_synchrony"])
        if "synchrony_energy" in ev:
            synchrony_energy.append(ev["synchrony_energy"])
        if "total_spikes" in ev:
            intensity.append(ev["total_spikes"])
        if prefix == "nb" and "fragment_count" in ev:
            fragment_counts.append(ev["fragment_count"])

    metrics[f"{prefix}_durations_list"] = json.dumps(durations)
    metrics[f"{prefix}_peak_list"] = json.dumps(peaks)
    metrics[f"{prefix}_intensity_list"] = json.dumps(intensity)
    metrics[f"{prefix}_synchrony_energy_list"] = json.dumps(synchrony_energy)
    metrics[f"{prefix}_fragment_count_list"] = json.dumps(fragment_counts)
    if prefix == "nb" and fragment_counts:
        metrics[f"{prefix}_fragment_count_mean"] = float(np.mean(fragment_counts))

    return metrics


def extract_metrics_from_json(path: str, root: Optional[str] = None, anchor: Optional[str] = ANCHOR) -> Dict[str, Any]:
    row: Dict[str, Any] = {"full_path": path}
    row.update(parse_path_metadata(path, root=root, anchor=anchor))

    try:
        with open(path, "r") as f:
            data = json.load(f)

        row["num_units"] = data.get("n_units", np.nan)
        row.update(extract_block_metrics(data.get("network_bursts"), "nb"))
        row.update(extract_block_metrics(data.get("superbursts"), "sb"))
        row.update(extract_block_metrics(data.get("burstlets"), "bl"))

    except Exception as e:
        row["error"] = str(e)
        row["num_units"] = np.nan

    return row


def main() -> int:
    args = parse_args()

    files = find_json_files(args.root, args.pattern)
    if not files:
        print(f"No files found under: {args.root}")
        return 1

    print(f"Found {len(files)} JSON files.")
    anchor = args.anchor or None
    records = [extract_metrics_from_json(path, root=args.root, anchor=anchor) for path in files]
    df = pd.DataFrame(records)

    # Keep important metadata columns first.
    preferred_order = [
        "Project",
        "Date",
        "Chip_ID",
        "RunID",
        "Well",
        "num_units",
        "full_path",
        "error",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[ordered_cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved CSV to: {output_path}")
    print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
