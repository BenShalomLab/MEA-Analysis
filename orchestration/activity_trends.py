#!/usr/bin/env python3
"""
Aggregate activity-scan results across sessions into longitudinal trends.

Individual scans answer "how is this chip doing today". This script answers the
questions that need several sessions:

  * How does activity mature over DIV?
  * Do experimental groups (genotypes, treatments) diverge, and when?
  * Is a chip or well degrading between timepoints?

Why scans are the right substrate for this
------------------------------------------
Network recordings use a *different* electrode selection each session, so
comparing their firing rates across timepoints quietly compares different
samples of the array. Activity scans cover the array on the same terms every
time, which makes them far more comparable longitudinally.

Input
-----
The ``summary.json`` files written by ``activity_scan.py`` — pass any directory
containing them at any depth.

DIV is taken from the recording's plating date and session date when both are
available; otherwise pass ``--div-from-path`` to parse it from folder names
(e.g. ``.../DIV14/...``), or fall back to session ordering.

Output
------
  trends.csv                tidy per-well-per-session table
  trends_by_group.png       group means over time, per metric
  trends_by_well.png        every well's trajectory, faint, with group means
  group_stats.json          per-timepoint group comparison (Mann-Whitney U)

Usage
-----
    python orchestration/activity_trends.py /data/scan_out
    python orchestration/activity_trends.py /data/scan_out --output-dir /data/trends
    python orchestration/activity_trends.py /data/scan_out \
        --metrics rate_mean_hz electrodes_active synchrony_fano
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOG = logging.getLogger("mea.activity_trends")

DEFAULT_METRICS = [
    "electrodes_active",
    "rate_mean_hz",
    "amplitude_mean_uv",
    "occupied_area_mm2",
    "synchrony_fano",
    "network_burst_rate_hz",
]

PRETTY = {
    "electrodes_active": "Active electrodes",
    "active_fraction": "Active fraction",
    "rate_mean_hz": "Mean firing rate (Hz)",
    "rate_median_hz": "Median firing rate (Hz)",
    "amplitude_mean_uv": "Mean amplitude (µV)",
    "occupied_area_mm2": "Active area (mm²)",
    "clustering_index": "Clustering index",
    "synchrony_fano": "Synchrony (Fano)",
    "network_burst_rate_hz": "Network burst rate (Hz)",
    "pct_spikes_in_bursts": "Spikes in bursts (%)",
    "isi_cv_median": "ISI CV (median)",
    "correlation_mean": "Mean correlation",
    "mean_degree": "Mean degree",
}

PALETTE = ["#4f46e5", "#eb6834", "#1baf7a", "#eda100", "#d55181", "#008300"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_date(text: str) -> Optional[date]:
    """Parse the date formats MaxWell and folder names use."""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%y%m%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def div_from_path(path: Path) -> Optional[int]:
    m = re.search(r"DIV[_\-]?(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def session_date_from_path(path: Path) -> Optional[date]:
    """Look for a YYMMDD or YYYYMMDD component in the path (e.g. .../240605/...)."""
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{6}", part):
            d = parse_date(part)
            if d:
                return d
        if re.fullmatch(r"\d{8}", part):
            d = parse_date(part)
            if d:
                return d
    return None


def load_summaries(root: Path) -> list[dict[str, Any]]:
    """Read every ``summary.json`` under ``root`` into flat per-well records."""
    rows: list[dict[str, Any]] = []
    files = sorted(root.rglob("summary.json"))
    LOG.info("Found %d summary file(s) under %s", len(files), root)

    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Skipping unreadable %s (%s)", fp, exc)
            continue

        source = Path(data.get("source", str(fp)))
        sess_date = session_date_from_path(source) or session_date_from_path(fp)
        path_div = div_from_path(source) or div_from_path(fp)

        for well in data.get("wells", []):
            plated = parse_date(well.get("plating_date", ""))
            div = path_div
            if div is None and plated and sess_date:
                div = (sess_date - plated).days

            rows.append({
                **well,
                "chip_id": data.get("chip_id", ""),
                "run_id": data.get("run_id", ""),
                "assay_type": data.get("assay_type", ""),
                "session_date": sess_date.isoformat() if sess_date else "",
                "div": div,
                "source": str(source),
                "summary_file": str(fp),
            })

    LOG.info("Loaded %d well record(s)", len(rows))
    return rows


def assign_timepoints(rows: list[dict]) -> str:
    """Pick the x-axis: DIV if we have it, otherwise ordered session dates.

    Returns the field name used as the timepoint.
    """
    if any(r.get("div") is not None for r in rows):
        for r in rows:
            r["timepoint"] = r.get("div")
        return "div"

    dates = sorted({r["session_date"] for r in rows if r.get("session_date")})
    if dates:
        order = {d: i for i, d in enumerate(dates)}
        for r in rows:
            r["timepoint"] = order.get(r.get("session_date"), None)
        LOG.warning("No DIV available — using session order as the timepoint")
        return "session"

    runs = sorted({r["run_id"] for r in rows})
    order = {r: i for i, r in enumerate(runs)}
    for r in rows:
        r["timepoint"] = order.get(r["run_id"], 0)
    LOG.warning("No DIV or session date — using run order as the timepoint")
    return "run"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mannwhitney_u(a: list[float], b: list[float]) -> Optional[float]:
    """Two-sided Mann-Whitney U p-value.

    Uses SciPy when available (exact for small n); otherwise a normal
    approximation with tie correction. Returns None if either group is too small
    to say anything — which is common with a handful of wells, and worth being
    honest about rather than reporting a meaningless number.
    """
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        pass

    x = np.asarray(a, float)
    y = np.asarray(b, float)
    combined = np.concatenate([x, y])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)

    # Average ranks within ties.
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()

    n1, n2 = x.size, y.size
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    ties = sum(c ** 3 - c for c in counts)
    n = n1 + n2
    sigma_sq = (n1 * n2 / 12) * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0
    if sigma_sq <= 0:
        return None
    z = (abs(u - mu) - 0.5) / np.sqrt(sigma_sq)
    return float(2 * 0.5 * np.erfc(z / np.sqrt(2))) if hasattr(np, "erfc") else \
        float(2 * (1 - _norm_cdf(z)))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math_erf(z / (2 ** 0.5)))


def math_erf(x: float) -> float:
    import math
    return math.erf(x)


def group_stats(rows: list[dict], metrics: list[str]) -> dict[str, Any]:
    """Compare groups at each timepoint, for each metric."""
    out: dict[str, Any] = {}
    timepoints = sorted({r["timepoint"] for r in rows if r.get("timepoint") is not None})
    groups = sorted({r.get("group", "") for r in rows if r.get("group")})
    if len(groups) < 2:
        return out

    for metric in metrics:
        per_tp = {}
        for tp in timepoints:
            vals = {g: [float(r[metric]) for r in rows
                        if r.get("timepoint") == tp and r.get("group") == g
                        and isinstance(r.get(metric), (int, float))]
                    for g in groups}
            entry: dict[str, Any] = {
                g: {"n": len(v),
                    "mean": round(float(np.mean(v)), 4) if v else None,
                    "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None}
                for g, v in vals.items()
            }
            if len(groups) == 2:
                a, b = vals[groups[0]], vals[groups[1]]
                p = mannwhitney_u(a, b)
                entry["comparison"] = {
                    "groups": groups,
                    "p_value": round(p, 5) if p is not None else None,
                    "note": None if p is not None else "too few wells per group to test",
                }
            per_tp[str(tp)] = entry
        out[metric] = per_tp
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _axis_label(kind: str) -> str:
    return {"div": "DIV", "session": "session (ordered)"}.get(kind, "run (ordered)")


def plot_group_trends(rows: list[dict], metrics: list[str], kind: str,
                      out: Path) -> Optional[Path]:
    """Group means over time, with SD bands and individual wells behind."""
    groups = sorted({r.get("group", "") for r in rows if r.get("group")}) or [""]
    tps = sorted({r["timepoint"] for r in rows if r.get("timepoint") is not None})
    if len(tps) < 2:
        LOG.warning("Only %d timepoint(s) — trend plot needs at least 2", len(tps))
        return None

    plt = _setup_mpl()
    cols = min(3, len(metrics))
    nrows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 3.3 * nrows),
                             squeeze=False, constrained_layout=True)

    for ax, metric in zip(axes.flat, metrics):
        for gi, g in enumerate(groups):
            colour = PALETTE[gi % len(PALETTE)]
            means, sds, xs = [], [], []
            for tp in tps:
                vals = [float(r[metric]) for r in rows
                        if r.get("timepoint") == tp and r.get("group") == g
                        and isinstance(r.get(metric), (int, float))]
                if vals:
                    xs.append(tp)
                    means.append(float(np.mean(vals)))
                    sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                    ax.scatter([tp] * len(vals), vals, s=13, color=colour,
                               alpha=0.35, linewidths=0, zorder=2)
            if xs:
                m, s = np.array(means), np.array(sds)
                ax.plot(xs, m, "-o", color=colour, lw=1.8, ms=4.5, label=g, zorder=3)
                ax.fill_between(xs, m - s, m + s, color=colour, alpha=0.13, lw=0)
        ax.set_title(PRETTY.get(metric, metric))
        ax.set_xlabel(_axis_label(kind))
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    if groups != [""]:
        axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Activity trends by experimental group (mean ± SD, wells shown individually)",
                 fontsize=10.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_well_trajectories(rows: list[dict], metrics: list[str], kind: str,
                           out: Path) -> Optional[Path]:
    """Every well as its own line — reveals individual wells that drop out."""
    tps = sorted({r["timepoint"] for r in rows if r.get("timepoint") is not None})
    if len(tps) < 2:
        return None
    plt = _setup_mpl()

    groups = sorted({r.get("group", "") for r in rows if r.get("group")}) or [""]
    colour_of = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    series: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r.get("timepoint") is None:
            continue
        key = (r.get("chip_id", ""), r.get("well_label", r.get("well_id")), r.get("group", ""))
        series[key][r["timepoint"]] = r

    cols = min(3, len(metrics))
    nrows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 3.3 * nrows),
                             squeeze=False, constrained_layout=True)

    for ax, metric in zip(axes.flat, metrics):
        for (chip, well, grp), by_tp in series.items():
            xs = sorted(t for t in by_tp if isinstance(by_tp[t].get(metric), (int, float)))
            if len(xs) < 2:
                continue
            ax.plot(xs, [float(by_tp[t][metric]) for t in xs], "-",
                    color=colour_of.get(grp, "#888"), alpha=0.45, lw=1)
        ax.set_title(PRETTY.get(metric, metric))
        ax.set_xlabel(_axis_label(kind))
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], color=colour_of[g], label=g) for g in groups if g]
    if handles:
        axes.flat[0].legend(handles=handles, fontsize=8, frameon=False)
    fig.suptitle("Per-well trajectories", fontsize=10.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_csv(rows: list[dict], metrics: list[str], out: Path) -> Path:
    cols = ["chip_id", "run_id", "session_date", "div", "timepoint",
            "well_id", "well_label", "group", "control", "qc", *metrics]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r.get("timepoint") or 0,
                                             r.get("chip_id", ""),
                                             str(r.get("well_label", "")))):
            w.writerow(r)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Directory containing activity-scan output")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write trends (default: <path>/trends)")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help="Metrics to track over time")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    rows = load_summaries(a.path)
    if not rows:
        raise SystemExit(f"No summary.json files found under {a.path}")

    kind = assign_timepoints(rows)
    out_dir = a.output_dir or (a.path / "trends")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only keep metrics that are actually present.
    metrics = [m for m in a.metrics
               if any(isinstance(r.get(m), (int, float)) for r in rows)]
    missing = set(a.metrics) - set(metrics)
    if missing:
        LOG.warning("Not present in these summaries, skipped: %s", ", ".join(sorted(missing)))
    if not metrics:
        raise SystemExit("None of the requested metrics are present in the summaries.")

    write_csv(rows, metrics, out_dir / "trends.csv")

    stats = group_stats(rows, metrics)
    if stats:
        (out_dir / "group_stats.json").write_text(json.dumps(stats, indent=2))

    if not a.no_figures:
        plot_group_trends(rows, metrics, kind, out_dir / "trends_by_group.png")
        plot_well_trajectories(rows, metrics, kind, out_dir / "trends_by_well.png")

    tps = sorted({r["timepoint"] for r in rows if r.get("timepoint") is not None})
    groups = sorted({r.get("group", "") for r in rows if r.get("group")})
    LOG.info("Wrote %s — %d wells, %d timepoint(s) (%s), groups: %s",
             out_dir, len(rows), len(tps), kind, ", ".join(groups) or "none")
    if len(tps) < 2:
        LOG.warning("Only one timepoint: trends need scans from multiple sessions.")


if __name__ == "__main__":
    main()
