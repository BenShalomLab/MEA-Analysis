#!/usr/bin/env python3
"""Collect raster plot files and build a day-arranged HTML dashboard."""

from __future__ import annotations

import argparse
import html
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_METADATA = {
    "project": None,
    "day": None,
    "chip_id": None,
    "run_id": None,
    "well": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a pipeline output tree for raster plot images and generate "
            "an HTML dashboard grouped by day."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory to scan (for example: AnalyzedData)",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("raster_dashboard.html"),
        help="Output HTML path (default: ./raster_dashboard.html)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="raster_burst_plot_60s.svg",
        help="Raster file name to collect recursively (default: raster_burst_plot_60s.svg)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="AnalyzedData",
        help="Directory name used as metadata anchor (default: AnalyzedData)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Raster Plot Dashboard",
        help="Dashboard page title",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        choices=tuple(DEFAULT_METADATA.keys()),
        default=["day"],
        help=(
            "Metadata field(s) used for dashboard sections. "
            "Default: day"
        ),
    )
    return parser.parse_args()


def empty_metadata() -> dict[str, Any]:
    return dict(DEFAULT_METADATA)


def parse_path_metadata(plot_path: Path, anchor: str) -> dict[str, Any]:
    """Extract path metadata from the output tree.

    Preferred parse uses an anchor like:
    .../<anchor>/.../<day>/<chip_id>/(Network/)<run_id>/<well>/<pattern>

    This supports both:
    .../<anchor>/<project>/<day>/<chip_id>/<run_id>/<well>/<pattern>
    .../<anchor>/<group>/<project>/<day>/<chip_id>/Network/<run_id>/<well>/<pattern>

    If anchor parsing is not possible, this falls back to:
    .../<run_id>/<well>/<pattern>
    """
    metadata: dict[str, Any] = empty_metadata()
    parts = list(plot_path.parts)

    if anchor in parts:
        idx = parts.index(anchor)
        body = parts[idx + 1 : -1]  # exclude anchor and filename

        if len(body) >= 2:
            metadata["well"] = body[-1]
            metadata["run_id"] = body[-2]

            # Optional Network folder between chip_id and run_id.
            chip_idx = -3
            if len(body) >= 3 and str(body[-3]).lower() == "network":
                chip_idx = -4

            if len(body) >= abs(chip_idx):
                metadata["chip_id"] = body[chip_idx]

            day_idx = chip_idx - 1
            if len(body) >= abs(day_idx):
                metadata["day"] = body[day_idx]

            # Everything before day is considered project context.
            if len(body) > abs(day_idx):
                project_tokens = body[:day_idx]
                metadata["project"] = "/".join(project_tokens) if project_tokens else None

            return metadata

    metadata["well"] = plot_path.parent.name if plot_path.parent else None
    metadata["run_id"] = plot_path.parent.parent.name if len(plot_path.parents) >= 2 else None

    # Best-effort day extraction from path tokens if anchor parse is unavailable.
    for token in parts:
        if re.fullmatch(r"(?i)day[_-]?\d+", token):
            metadata["day"] = token
            break
        if re.fullmatch(r"\d{6,8}", token):
            metadata["day"] = token
            break

    return metadata


def day_sort_key(day_value: Any) -> tuple[int, str]:
    day_str = "Unknown" if day_value in (None, "") else str(day_value)
    match = re.search(r"(\d+)", day_str)
    if match:
        return (0, f"{int(match.group(1)):08d}")
    return (1, day_str.lower())


def format_group_label(row: dict[str, Any], group_by: list[str]) -> str:
    values = []
    for field in group_by:
        value = row.get(field) or "Unknown"
        values.append(f"{field}={value}")
    return " | ".join(values)


def group_sort_key(section_label: str) -> tuple[Any, ...]:
    keys: list[Any] = []
    for part in section_label.split(" | "):
        _, _, value = part.partition("=")
        if part.startswith("day="):
            keys.append(day_sort_key(value))
        else:
            keys.append((0, str(value).lower()))
    return tuple(keys)


def render_dashboard(rows: list[dict[str, Any]], title: str, group_by: list[str]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        section = format_group_label(row, group_by)
        grouped[section].append(row)

    sections: list[str] = []
    for section_label in sorted(grouped.keys(), key=group_sort_key):
        cards: list[str] = []
        for row in sorted(grouped[section_label], key=lambda r: ((r.get("project") or ""), (r.get("day") or ""), (r.get("chip_id") or ""), (r.get("run_id") or ""), (r.get("well") or ""))):
            label = " | ".join(
                x
                for x in [
                    row.get("project"),
                    row.get("day"),
                    row.get("chip_id"),
                    row.get("run_id"),
                    row.get("well"),
                ]
                if x
            ) or "Unlabeled"

            cards.append(
                """
                <article class=\"card\"> 
                  <div class=\"card-title\">{label}</div>
                  <a href=\"{img}\" target=\"_blank\" rel=\"noopener noreferrer\">
                    <img loading=\"lazy\" src=\"{img}\" alt=\"{alt}\" />
                  </a>
                  <div class=\"path\">{path}</div>
                </article>
                """.format(
                    label=html.escape(label),
                    img=html.escape(row["image_rel_path"]),
                    alt=html.escape(label),
                    path=html.escape(row["image_rel_path"]),
                )
            )

        sections.append(
            """
            <section class=\"day-section\">
              <h2>{section_label}</h2>
              <div class=\"grid\">{cards}</div>
            </section>
            """.format(section_label=html.escape(section_label), cards="".join(cards))
        )

    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 1rem 1.25rem 2rem; background: #fafafa; color: #222; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .meta {{ color: #555; margin-bottom: 1.25rem; }}
    .day-section {{ margin-top: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 0.9rem; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 0.6rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .card-title {{ font-size: 0.85rem; margin-bottom: 0.35rem; font-weight: 600; }}
    img {{ width: 100%; height: 220px; object-fit: contain; border: 1px solid #eee; background: #fff; }}
    .path {{ margin-top: 0.35rem; font-size: 0.72rem; color: #666; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class=\"meta\">Total plots: {count}</div>
  {sections}
</body>
</html>
""".format(title=html.escape(title), count=len(rows), sections="".join(sections))


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_html = args.output_html.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    plot_paths = sorted(input_dir.rglob(args.pattern))
    if not plot_paths:
        raise SystemExit(f"No {args.pattern} files found under: {input_dir}")

    rows: list[dict[str, Any]] = []
    for plot_path in plot_paths:
        row = parse_path_metadata(plot_path, args.anchor)
        row["image_abs_path"] = str(plot_path)
        row["image_rel_path"] = str(plot_path.relative_to(output_html.parent))
        rows.append(row)

    html_text = render_dashboard(rows, args.title, args.group_by)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")

    print(f"Wrote dashboard with {len(rows)} plots to {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
