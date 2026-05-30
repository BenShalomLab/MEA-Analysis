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
    return parser.parse_args()


def empty_metadata() -> dict[str, Any]:
    return dict(DEFAULT_METADATA)


def parse_path_metadata(plot_path: Path, anchor: str) -> dict[str, Any]:
    """Extract path metadata from the output tree.

    Preferred parse:
    .../<anchor>/<project>/<day>/<chip_id>/<run_id>/<well>/<pattern>

    Falls back to:
    .../<run_id>/<well>/<pattern>
    """
    metadata: dict[str, Any] = empty_metadata()
    parts = list(plot_path.parts)

    if anchor in parts:
        idx = parts.index(anchor)
        try:
            metadata["project"] = parts[idx + 1]
            metadata["day"] = parts[idx + 2]
            metadata["chip_id"] = parts[idx + 3]
            metadata["run_id"] = parts[idx + 4]
            metadata["well"] = parts[idx + 5]
            return metadata
        except IndexError:
            pass

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


def render_dashboard(rows: list[dict[str, Any]], title: str) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        day = row.get("day") or "Unknown"
        grouped[str(day)].append(row)

    day_sections: list[str] = []
    for day in sorted(grouped.keys(), key=day_sort_key):
        cards: list[str] = []
        for row in sorted(grouped[day], key=lambda r: ((r.get("project") or ""), (r.get("chip_id") or ""), (r.get("run_id") or ""), (r.get("well") or ""))):
            label = " | ".join(
                x
                for x in [
                    row.get("project"),
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

        day_sections.append(
            """
            <section class=\"day-section\">
              <h2>{day}</h2>
              <div class=\"grid\">{cards}</div>
            </section>
            """.format(day=html.escape(day), cards="".join(cards))
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
""".format(title=html.escape(title), count=len(rows), sections="".join(day_sections))


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

    html_text = render_dashboard(rows, args.title)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")

    print(f"Wrote dashboard with {len(rows)} plots to {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
