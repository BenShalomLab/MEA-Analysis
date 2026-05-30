#!/usr/bin/env python3
"""Collect network_results.json files into a single CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import chain
from pathlib import Path
from typing import Any


SCALAR_TYPES = (str, int, float, bool, type(None))
DEFAULT_METADATA = {
    "project": None,
    "date": None,
    "chip_id": None,
    "run_id": None,
    "well": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a pipeline output tree for network_results.json files and "
            "write one consolidated CSV for downstream plotting."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory to scan (for example: AnalyzedData)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("network_results_summary.csv"),
        help="Output CSV path (default: ./network_results_summary.csv)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="AnalyzedData",
        help="Directory name used as metadata anchor (default: AnalyzedData)",
    )
    return parser.parse_args()


def empty_metadata() -> dict[str, Any]:
    return dict(DEFAULT_METADATA)


def parse_path_metadata(json_path: Path, anchor: str) -> dict[str, Any]:
    """Extract path metadata from the output tree.

    Preferred parse uses an anchor like:
    .../<anchor>/.../<date>/<chip_id>/(Network/)<run_id>/<well>/network_results.json

    This supports both:
    .../<anchor>/<project>/<date>/<chip_id>/<run_id>/<well>/network_results.json
    .../<anchor>/<group>/<project>/<date>/<chip_id>/Network/<run_id>/<well>/network_results.json

    If anchor parsing is not possible, this falls back to:
    .../<run_id>/<well>/network_results.json
    """
    parts = list(json_path.parts)
    metadata: dict[str, Any] = empty_metadata()

    if anchor in parts:
        idx = parts.index(anchor)
        body = parts[idx + 1 : -1]  # exclude anchor and filename

        if len(body) >= 2:
            metadata["well"] = body[-1]
            metadata["run_id"] = body[-2]

            # Optional assay folder between chip_id and run_id.
            assay_idx = -3
            if len(body) >= 3 and str(body[-3]).lower() == "network":
                assay_idx = -4

            if len(body) >= abs(assay_idx):
                metadata["chip_id"] = body[assay_idx]

            date_idx = assay_idx - 1
            if len(body) >= abs(date_idx):
                metadata["date"] = body[date_idx]

            # Everything before date is considered project context.
            if len(body) > abs(date_idx):
                project_tokens = body[:date_idx]
                metadata["project"] = "/".join(project_tokens) if project_tokens else None

            return metadata

    # Fallback to local parent structure: .../<run>/<well>/network_results.json
    if len(json_path.parents) >= 2:
        metadata["well"] = json_path.parent.name
        metadata["run_id"] = json_path.parent.parent.name

    return metadata


def flatten_scalars(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}_{key}" if prefix else key
            flat.update(flatten_scalars(value, next_prefix))
        return flat

    if isinstance(data, SCALAR_TYPES):
        flat[prefix] = data

    return flat


def extract_event_counts(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for block_name in ("burstlets", "network_bursts", "superbursts"):
        block = data.get(block_name)
        key = f"{block_name}_event_count"

        if isinstance(block, dict):
            events = block.get("events", [])
            out[key] = len(events) if isinstance(events, list) else 0
        elif isinstance(block, list):
            out[key] = len(block)
        else:
            out[key] = 0

    return out


def load_row(json_path: Path, anchor: str) -> dict[str, Any]:
    row: dict[str, Any] = {"json_path": str(json_path)}
    row.update(parse_path_metadata(json_path, anchor))

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        row["error"] = "json_root_not_dict"
        return row

    flat = flatten_scalars(data)
    row.update(flat)
    row.update(extract_event_counts(data))
    return row


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = list(dict.fromkeys(chain.from_iterable(row.keys() for row in rows)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_csv = args.output_csv.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    json_paths = sorted(input_dir.rglob("network_results.json"))
    if not json_paths:
        raise SystemExit(f"No network_results.json files found under: {input_dir}")

    rows: list[dict[str, Any]] = []
    for path in json_paths:
        try:
            rows.append(load_row(path, args.anchor))
        except Exception as exc:
            rows.append(
                {
                    "json_path": str(path),
                    **empty_metadata(),
                    "error": str(exc),
                }
            )

    write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
