#!/usr/bin/env python3
"""
Extract whole-array activity information from MaxWell recordings.

Motivation
----------
A MaxOne/MaxTwo chip has ~26,400 electrodes but can only record ~1,020 at once.
The ActivityScan sweeps the whole array in blocks to find where the activity is;
that map then selects the ~1,020 electrodes used for the Network recording. So
the Network file you spike-sort sees roughly 4% of the chip — the ActivityScan
is the only view of the rest, and it is normally discarded.

This module reads it and produces per-electrode maps, per-well quality metrics,
figures, and machine-readable summaries.

Why this is cheap
-----------------
MaxWell stores online threshold-crossing spikes in ``spikes`` datasets as
``(frameno, channel, amplitude)``, alongside a ``settings/mapping`` table giving
each channel's electrode id and x/y position in micrometres. Both are plain,
uncompressed HDF5 — so **no spike sorting, no GPU, and no MaxWell HDF5
compression plugin are required**. Only the raw voltage traces need the plugin,
and we never touch them.

What it produces
----------------
Per run, under ``<output>/<chip>/<run_id>/``:

  summary.json              per-well metrics, conditions, and scan parameters
  per_electrode.csv         electrode id, x, y, spikes, duration, rate, amplitude
  well<NNN>_activity.png    whole-array firing-rate and amplitude maps
  plate_overview.png        all wells side by side on a common scale

Usage
-----
    # One file
    python orchestration/activity_scan.py /path/to/ActivityScan/000170/data.raw.h5

    # A whole session tree (finds every data.raw.h5 under ActivityScan folders)
    python orchestration/activity_scan.py /data/240605 --output-dir /data/scan_out

    # Include the Network selection overlay, to show which electrodes were kept
    python orchestration/activity_scan.py /data/240605/M06804/ActivityScan/000170 \
        --selection-from /data/240605/M06804/Network/000175/data.raw.h5

    # Metrics only, no figures (fast)
    python orchestration/activity_scan.py <path> --no-figures
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("h5py is required:  pip install h5py")

LOG = logging.getLogger("mea.activity_scan")

# MaxWell full-array geometry (MaxOne / MaxTwo HD-MEA)
ARRAY_ELECTRODES = 26_400
ELECTRODE_PITCH_UM = 17.5


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _s(value: Any) -> str:
    """Decode an HDF5 scalar that may be bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size:
        return _s(value.flat[0])
    return str(value)


def _first(dset) -> Any:
    try:
        return dset[0]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class WellMeta:
    """Plate-level description of a well (condition labels live here)."""
    well_id: int
    label: str = ""            # printed well name, e.g. "1"
    group: str = ""            # experimental group / genotype, e.g. "MxWT"
    color: str = ""
    control: bool = False
    plating_date: str = ""


@dataclass
class WellActivity:
    """Aggregated whole-array activity for one well."""
    well_id: int
    meta: WellMeta

    electrode: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    spikes: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    seconds: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    amp_sum: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))

    blocks: int = 0
    scan_seconds: float = 0.0
    sampling_hz: float = 0.0
    spike_threshold: float = 0.0

    @property
    def rate(self) -> np.ndarray:
        """Per-electrode firing rate (Hz).

        Each electrode is divided by the time *it* was actually routed, which
        differs between electrodes in a multi-block scan.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(self.seconds > 0, self.spikes / self.seconds, 0.0)
        return np.nan_to_num(r)

    @property
    def mean_amplitude(self) -> np.ndarray:
        """Per-electrode mean absolute spike amplitude (µV)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(self.spikes > 0, self.amp_sum / np.maximum(self.spikes, 1), 0.0)
        return np.nan_to_num(a)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_wellplate(f: h5py.File) -> dict[int, WellMeta]:
    """Well labels and experimental groups from the ``wellplate`` group."""
    out: dict[int, WellMeta] = {}
    if "wellplate" not in f:
        return out
    for key in f["wellplate"]:
        grp = f["wellplate"][key]
        if not isinstance(grp, h5py.Group):
            continue
        try:
            wid = int(_first(grp["id"]))
        except Exception:  # noqa: BLE001
            continue
        out[wid] = WellMeta(
            well_id=wid,
            label=_s(_first(grp["name"])) if "name" in grp else "",
            group=_s(_first(grp["group_name"])) if "group_name" in grp else "",
            color=_s(_first(grp["group_color"])) if "group_color" in grp else "",
            control=bool(_first(grp["control"])) if "control" in grp else False,
            plating_date=_s(_first(grp["Plating Date"])) if "Plating Date" in grp else "",
        )
    return out


def iter_blocks(f: h5py.File) -> Iterator[tuple[int, h5py.Group]]:
    """Yield ``(well_id, block)`` for every recording block in the file.

    ``data_store/dataNNNN`` holds one entry per recording x well, which is the
    flat view we want: an ActivityScan has many blocks per well (one per
    electrode configuration), a Network recording usually has one.
    """
    if "data_store" not in f:
        return
    for key in sorted(f["data_store"]):
        blk = f["data_store"][key]
        if not isinstance(blk, h5py.Group) or "spikes" not in blk:
            continue
        try:
            wid = int(_first(blk["well_id"]))
        except Exception:  # noqa: BLE001
            continue
        yield wid, blk


def block_duration_s(blk: h5py.Group) -> float:
    """Recording length of a block, in seconds (times are epoch milliseconds)."""
    try:
        start, stop = int(_first(blk["start_time"])), int(_first(blk["stop_time"]))
        if stop > start:
            return (stop - start) / 1000.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def read_selection(path: Path) -> dict[int, set[int]]:
    """Electrodes chosen for recording, from ``assay/inputs/electrodes``.

    Returns ``{well_id: {electrode_id, ...}}``. Used to overlay the Network
    selection on the scan map, showing which part of the activity was kept.
    """
    out: dict[int, set[int]] = {}
    try:
        with h5py.File(path, "r") as f:
            if "assay/inputs/electrodes" not in f:
                return out
            payload = json.loads(_s(_first(f["assay/inputs/electrodes"])))
            for well, elecs in (payload.get("electrodes") or {}).items():
                out[int(well)] = {int(e) for e in elecs}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not read electrode selection from %s: %s", path, exc)
    return out


def extract_file(path: Path, max_spikes_per_block: int = 0) -> dict[int, WellActivity]:
    """Aggregate per-electrode activity for every well in one HDF5 file.

    Spike counts and routed time accumulate across all blocks, so a scan that
    visits an electrode in several configurations is handled correctly.
    """
    wells: dict[int, WellActivity] = {}

    with h5py.File(path, "r") as f:
        plate = read_wellplate(f)

        # electrode id -> accumulated (spikes, seconds, |amplitude| sum, x, y)
        acc: dict[int, dict[int, list[float]]] = {}

        for wid, blk in iter_blocks(f):
            if "settings/mapping" not in blk:
                continue
            mapping = blk["settings/mapping"][:]
            duration = block_duration_s(blk)

            wa = wells.get(wid)
            if wa is None:
                wa = WellActivity(well_id=wid, meta=plate.get(wid, WellMeta(well_id=wid)))
                wells[wid] = wa
                acc[wid] = {}
            wa.blocks += 1
            wa.scan_seconds += duration
            if not wa.sampling_hz and "settings/sampling" in blk:
                wa.sampling_hz = float(_first(blk["settings/sampling"]) or 0)
            if not wa.spike_threshold and "settings/spike_threshold" in blk:
                wa.spike_threshold = float(_first(blk["settings/spike_threshold"]) or 0)

            # channel -> electrode/x/y for this configuration
            chan = mapping["channel"].astype(np.int64)
            elec = mapping["electrode"].astype(np.int64)
            xs, ys = mapping["x"].astype(np.float64), mapping["y"].astype(np.float64)
            n_chan = int(chan.max()) + 1 if chan.size else 0

            chan_to_idx = np.full(n_chan, -1, np.int64)
            chan_to_idx[chan] = np.arange(chan.size)

            # Register every routed electrode, even if it never spiked — a silent
            # electrode is a real observation, not a missing one.
            store = acc[wid]
            for i in range(elec.size):
                rec = store.get(int(elec[i]))
                if rec is None:
                    store[int(elec[i])] = [0.0, duration, 0.0, float(xs[i]), float(ys[i])]
                else:
                    rec[1] += duration

            sp = blk["spikes"]
            if sp.shape[0]:
                take = slice(0, max_spikes_per_block) if max_spikes_per_block else slice(None)
                data = sp[take]
                sch = data["channel"].astype(np.int64)
                amp = np.abs(data["amplitude"].astype(np.float64))

                valid = (sch >= 0) & (sch < n_chan)
                sch, amp = sch[valid], amp[valid]
                idx = chan_to_idx[sch]
                ok = idx >= 0
                idx, amp = idx[ok], amp[ok]

                counts = np.bincount(idx, minlength=elec.size)
                amps = np.bincount(idx, weights=amp, minlength=elec.size)
                hit = np.nonzero(counts)[0]
                for i in hit:
                    rec = store[int(elec[i])]
                    rec[0] += float(counts[i])
                    rec[2] += float(amps[i])

        # Freeze the accumulators into arrays, ordered by electrode id.
        for wid, wa in wells.items():
            store = acc[wid]
            if not store:
                continue
            ids = np.array(sorted(store), dtype=np.int64)
            rows = np.array([store[int(e)] for e in ids], dtype=np.float64)
            wa.electrode = ids
            wa.spikes = rows[:, 0].astype(np.int64)
            wa.seconds = rows[:, 1]
            wa.amp_sum = rows[:, 2]
            wa.x = rows[:, 3]
            wa.y = rows[:, 4]

    return wells


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def well_metrics(wa: WellActivity, active_hz: float = 0.05,
                 selection: Optional[set[int]] = None) -> dict[str, Any]:
    """Quality and activity metrics for one well.

    ``active_hz`` is the firing-rate threshold for calling an electrode active;
    0.05 Hz matches the pipeline's own default firing-rate curation threshold.
    """
    rate = wa.rate
    amp = wa.mean_amplitude
    n = int(rate.size)
    active = rate >= active_hz
    n_active = int(active.sum())

    m: dict[str, Any] = {
        "well_id": wa.well_id,
        "well_label": wa.meta.label,
        "group": wa.meta.group,
        "control": wa.meta.control,
        "plating_date": wa.meta.plating_date,
        "blocks": wa.blocks,
        "scan_seconds": round(wa.scan_seconds, 1),
        "sampling_hz": wa.sampling_hz,
        "spike_threshold": wa.spike_threshold,
        "electrodes_scanned": n,
        "array_coverage_pct": round(100 * n / ARRAY_ELECTRODES, 1) if n else 0.0,
        "electrodes_active": n_active,
        "active_fraction": round(n_active / n, 4) if n else 0.0,
        "total_spikes": int(wa.spikes.sum()),
    }

    if n_active:
        ra, aa = rate[active], amp[active]
        m.update({
            "rate_mean_hz": round(float(ra.mean()), 4),
            "rate_median_hz": round(float(np.median(ra)), 4),
            "rate_p90_hz": round(float(np.percentile(ra, 90)), 4),
            "rate_max_hz": round(float(ra.max()), 4),
            "amplitude_mean_uv": round(float(aa.mean()), 2),
            "amplitude_median_uv": round(float(np.median(aa)), 2),
            "amplitude_p90_uv": round(float(np.percentile(aa, 90)), 2),
        })

        # Spatial organisation of the active population.
        ax, ay = wa.x[active], wa.y[active]
        cx, cy = float(ax.mean()), float(ay.mean())
        m["centroid_um"] = [round(cx, 1), round(cy, 1)]
        m["dispersion_um"] = round(float(np.sqrt(((ax - cx) ** 2 + (ay - cy) ** 2).mean())), 1)

        # Occupied area, on a coarse grid — robust to single stray electrodes.
        bin_um = 100.0
        gx = np.floor(ax / bin_um).astype(np.int64)
        gy = np.floor(ay / bin_um).astype(np.int64)
        occupied = len(set(zip(gx.tolist(), gy.tolist())))
        m["occupied_bins_100um"] = occupied
        m["occupied_area_mm2"] = round(occupied * (bin_um / 1000.0) ** 2, 3)

        # Clustering: how often an active electrode sits next to another one.
        # 1.0 = fully contiguous tissue, near 0 = isolated scattered units.
        act_set = set(zip(np.round(ax / ELECTRODE_PITCH_UM).astype(int).tolist(),
                          np.round(ay / ELECTRODE_PITCH_UM).astype(int).tolist()))
        neighbours = 0
        for gxi, gyi in act_set:
            if any((gxi + dx, gyi + dy) in act_set
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                neighbours += 1
        m["clustering_index"] = round(neighbours / len(act_set), 3) if act_set else 0.0
    else:
        m.update({k: 0.0 for k in (
            "rate_mean_hz", "rate_median_hz", "rate_p90_hz", "rate_max_hz",
            "amplitude_mean_uv", "amplitude_median_uv", "amplitude_p90_uv",
        )})

    # How the electrodes actually recorded compare with the whole array.
    if selection:
        sel_mask = np.isin(wa.electrode, list(selection))
        n_sel = int(sel_mask.sum())
        m["selected_electrodes"] = n_sel
        if n_sel:
            sel_rate = rate[sel_mask]
            m["selected_rate_mean_hz"] = round(float(sel_rate.mean()), 4)
            m["selected_active_fraction"] = round(float((sel_rate >= active_hz).mean()), 4)
            if n_active:
                # >1 means the selection is enriched for active electrodes,
                # which is the point — but it also biases downstream rates.
                m["selection_enrichment"] = round(
                    float(sel_rate.mean() / max(rate.mean(), 1e-9)), 2)
    return m


def quality_flag(m: dict[str, Any], min_active: int = 50,
                 min_rate: float = 0.1) -> tuple[str, list[str]]:
    """Coarse pass/warn/fail verdict, for gating expensive downstream analysis."""
    reasons: list[str] = []
    if m["electrodes_active"] < min_active:
        reasons.append(f"only {m['electrodes_active']} active electrodes")
    if m.get("rate_mean_hz", 0) < min_rate:
        reasons.append(f"low mean rate ({m.get('rate_mean_hz', 0)} Hz)")
    if m["active_fraction"] < 0.01:
        reasons.append(f"active fraction {m['active_fraction']:.1%}")
    if not reasons:
        return "pass", []
    return ("fail" if m["electrodes_active"] < min_active // 2 else "warn"), reasons


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


def plot_well(wa: WellActivity, metrics: dict, out: Path,
              selection: Optional[set[int]] = None, active_hz: float = 0.05) -> Optional[Path]:
    """Whole-array firing-rate map, amplitude map, and rate distribution."""
    if wa.electrode.size == 0:
        return None
    plt = _setup_mpl()

    rate, amp = wa.rate, wa.mean_amplitude
    active = rate >= active_hz

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})

    # 1. Firing-rate map. Silent electrodes stay visible in grey so coverage is
    #    distinguishable from inactivity.
    ax = axes[0]
    ax.scatter(wa.x[~active], wa.y[~active], s=1.2, c="#e0e0e0", linewidths=0, label="silent")
    if active.any():
        sc = ax.scatter(wa.x[active], wa.y[active], s=3.2, c=rate[active],
                        cmap="viridis", linewidths=0,
                        vmin=0, vmax=float(np.percentile(rate[active], 99)) or 1)
        fig.colorbar(sc, ax=ax, label="firing rate (Hz)", fraction=0.046, pad=0.02)
    if selection:
        sel = np.isin(wa.electrode, list(selection))
        # Only worth drawing when the selection is a genuine subset; if nearly
        # everything scanned was also recorded, the overlay just obscures the map.
        if sel.any() and sel.mean() < 0.9:
            ax.scatter(wa.x[sel], wa.y[sel], s=11, facecolors="none",
                       edgecolors="#d62728", linewidths=0.35, label="recorded")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      ncol=2, fontsize=7.5, frameon=False)
    ax.set_title(f"Activity map — well {wa.meta.label or wa.well_id}"
                 + (f" ({wa.meta.group})" if wa.meta.group else ""))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.invert_yaxis()

    # 2. Amplitude map — proxy for signal quality / proximity to a soma.
    ax = axes[1]
    ax.scatter(wa.x[~active], wa.y[~active], s=1.2, c="#e0e0e0", linewidths=0)
    if active.any():
        sc = ax.scatter(wa.x[active], wa.y[active], s=3.2, c=amp[active],
                        cmap="magma", linewidths=0,
                        vmin=0, vmax=float(np.percentile(amp[active], 99)) or 1)
        fig.colorbar(sc, ax=ax, label="mean |amplitude| (µV)", fraction=0.046, pad=0.02)
    ax.set_title("Spike amplitude")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.invert_yaxis()

    # 3. Rate distribution, with the active threshold marked.
    ax = axes[2]
    if active.any():
        ax.hist(np.log10(rate[active]), bins=40, color="#4f46e5", alpha=0.85)
        ax.set_xlabel("log₁₀ firing rate (Hz)")
        ax.set_ylabel("electrodes")
        ax.axvline(math.log10(active_hz), color="#d62728", ls="--", lw=1,
                   label=f"active ≥ {active_hz} Hz")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("Rate distribution")

    q, _ = quality_flag(metrics)
    fig.suptitle(
        f"{metrics['electrodes_active']:,} active / {metrics['electrodes_scanned']:,} scanned "
        f"({metrics['active_fraction']:.1%})   ·   mean {metrics.get('rate_mean_hz', 0)} Hz   ·   "
        f"area {metrics.get('occupied_area_mm2', 0)} mm²   ·   QC: {q}",
        fontsize=9, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_plate(wells: dict[int, WellActivity], metrics: dict[int, dict],
               out: Path, active_hz: float = 0.05) -> Optional[Path]:
    """All wells on one shared colour scale, for at-a-glance comparison."""
    if not wells:
        return None
    plt = _setup_mpl()

    ordered = [wells[k] for k in sorted(wells)]
    # 95th percentile of the pooled active population: high enough to resist
    # outliers, low enough that typical electrodes are not all crushed to black.
    pooled = np.concatenate([w.rate[w.rate >= active_hz] for w in ordered
                             if (w.rate >= active_hz).any()]) if ordered else np.empty(0)
    peak = float(np.percentile(pooled, 95)) if pooled.size else 1.0
    peak = peak or 1.0

    cols = min(len(ordered), 3)
    rows = math.ceil(len(ordered) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows),
                             squeeze=False, constrained_layout=True)

    sc = None
    for ax, wa in zip(axes.flat, ordered):
        rate = wa.rate
        active = rate >= active_hz
        m = metrics.get(wa.well_id, {})
        ax.scatter(wa.x[~active], wa.y[~active], s=1.0, c="#ececec", linewidths=0)
        if active.any():
            sc = ax.scatter(wa.x[active], wa.y[active], s=2.6, c=rate[active],
                            cmap="viridis", vmin=0, vmax=peak, linewidths=0)
        q, _ = quality_flag(m) if m else ("", [])
        colour = {"pass": "#079455", "warn": "#b54708", "fail": "#d92d20"}.get(q, "#475467")
        ax.set_title(f"well {wa.meta.label or wa.well_id}"
                     + (f" · {wa.meta.group}" if wa.meta.group else "")
                     + f"\n{m.get('electrodes_active', 0):,} active · {q}",
                     fontsize=8.5, color=colour, pad=6)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(ordered):]:
        ax.axis("off")

    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="firing rate (Hz)",
                     fraction=0.02, pad=0.01)
    fig.suptitle("Whole-array activity by well (shared scale)", fontsize=10.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_groups(metrics: dict[int, dict], out: Path) -> Optional[Path]:
    """Compare experimental groups (genotype/condition) on key scan metrics.

    Groups come from ``wellplate/well*/group_name`` in the recording itself, so
    no external plate map is needed. Individual wells are drawn as points — with
    a handful of wells per group, the spread matters more than the mean.
    """
    groups: dict[str, list[dict]] = {}
    for m in metrics.values():
        if m.get("group"):
            groups.setdefault(m["group"], []).append(m)
    if len(groups) < 2:
        return None

    plt = _setup_mpl()
    fields = [
        ("electrodes_active", "Active electrodes"),
        ("rate_mean_hz", "Mean firing rate (Hz)"),
        ("amplitude_mean_uv", "Mean amplitude (µV)"),
        ("occupied_area_mm2", "Active area (mm²)"),
    ]
    names = sorted(groups)
    palette = ["#4f46e5", "#eb6834", "#1baf7a", "#eda100", "#d55181"]

    fig, axes = plt.subplots(1, len(fields), figsize=(3.3 * len(fields), 3.4))
    for ax, (key, label) in zip(np.atleast_1d(axes), fields):
        for i, g in enumerate(names):
            vals = [m.get(key, 0) or 0 for m in groups[g]]
            colour = palette[i % len(palette)]
            ax.bar(i, float(np.mean(vals)), 0.6, color=colour, alpha=0.35,
                   edgecolor=colour, linewidth=1.2)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)),
                       vals, s=22, color=colour, zorder=3,
                       edgecolors="white", linewidths=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_title(label, fontsize=9)
        ax.margins(x=0.25)

    n_txt = ", ".join(f"{g} n={len(groups[g])}" for g in names)
    fig.suptitle(f"Activity by experimental group   ({n_txt})", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_per_electrode_csv(wells: dict[int, WellActivity], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["well_id", "well_label", "group", "electrode", "x_um", "y_um",
                    "spikes", "seconds", "rate_hz", "mean_amplitude_uv"])
        for wid in sorted(wells):
            wa = wells[wid]
            rate, amp = wa.rate, wa.mean_amplitude
            for i in range(wa.electrode.size):
                w.writerow([wid, wa.meta.label, wa.meta.group, int(wa.electrode[i]),
                            round(float(wa.x[i]), 1), round(float(wa.y[i]), 1),
                            int(wa.spikes[i]), round(float(wa.seconds[i]), 1),
                            round(float(rate[i]), 5), round(float(amp[i]), 2)])
    return out


def process_file(h5_path: Path, out_dir: Path, *, figures: bool = True,
                 active_hz: float = 0.05, selection_from: Optional[Path] = None,
                 max_spikes_per_block: int = 0) -> dict[str, Any]:
    """Extract, measure, plot, and write everything for a single recording."""
    LOG.info("Reading %s", h5_path)
    wells = extract_file(h5_path, max_spikes_per_block=max_spikes_per_block)
    if not wells:
        LOG.warning("No readable recording blocks in %s", h5_path)
        return {}

    selection = read_selection(selection_from) if selection_from else {}

    run_id, chip_id, script_id = h5_path.parent.name, "", ""
    try:
        with h5py.File(h5_path, "r") as f:
            if "assay/run_id" in f:
                run_id = _s(_first(f["assay/run_id"])) or run_id
            if "assay/script_id" in f:
                script_id = _s(_first(f["assay/script_id"]))
            if "wellplate/id" in f:
                chip_id = _s(_first(f["wellplate/id"]))
    except Exception:  # noqa: BLE001
        pass

    dest = out_dir / (chip_id or "unknown_chip") / run_id
    dest.mkdir(parents=True, exist_ok=True)

    metrics: dict[int, dict] = {}
    for wid, wa in sorted(wells.items()):
        m = well_metrics(wa, active_hz=active_hz, selection=selection.get(wid))
        verdict, reasons = quality_flag(m)
        m["qc"] = verdict
        m["qc_reasons"] = reasons
        metrics[wid] = m
        LOG.info("  well %s (%s): %s active / %s scanned (%.1f%%), mean %.3f Hz — %s",
                 wa.meta.label or wid, wa.meta.group or "?",
                 f"{m['electrodes_active']:,}", f"{m['electrodes_scanned']:,}",
                 100 * m["active_fraction"], m.get("rate_mean_hz", 0), verdict)

    summary = {
        "source": str(h5_path),
        "chip_id": chip_id,
        "run_id": run_id,
        "script_id": script_id,
        "assay_type": ("activity_scan" if "activity" in script_id.lower()
                       else "network" if "network" in script_id.lower() else "unknown"),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "active_threshold_hz": active_hz,
        "array_electrodes": ARRAY_ELECTRODES,
        "wells": [metrics[k] for k in sorted(metrics)],
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    write_per_electrode_csv(wells, dest / "per_electrode.csv")

    if figures:
        for wid, wa in sorted(wells.items()):
            plot_well(wa, metrics[wid], dest / f"well{wid:03d}_activity.png",
                      selection=selection.get(wid), active_hz=active_hz)
        plot_plate(wells, metrics, dest / "plate_overview.png", active_hz=active_hz)
        plot_groups(metrics, dest / "group_comparison.png")

    LOG.info("  wrote %s", dest)
    summary["output_dir"] = str(dest)
    return summary


# --------------------------------------------------------------------------- #
# Discovery + CLI
# --------------------------------------------------------------------------- #
def discover(path: Path, assay_subfolder: Optional[str] = "ActivityScan",
             h5_glob: str = "data.raw.h5") -> list[Path]:
    """Find recordings to process, mirroring the pipeline's own conventions."""
    if path.is_file():
        return [path]
    hits = sorted(path.rglob(h5_glob))
    if assay_subfolder:
        filtered = [p for p in hits if assay_subfolder in p.parts]
        if filtered:
            return filtered
        LOG.warning("No recordings under a '%s' folder; using all %d found",
                    assay_subfolder, len(hits))
    return hits


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path,
                   help="data.raw.h5, or a directory to search")
    p.add_argument("--output-dir", type=Path, default=Path("activity_scan_output"))
    p.add_argument("--assay-subfolder", default="ActivityScan",
                   help="Only process recordings under this folder (blank = any)")
    p.add_argument("--active-hz", type=float, default=0.05,
                   help="Firing rate above which an electrode counts as active")
    p.add_argument("--selection-from", type=Path, default=None,
                   help="Network data.raw.h5 whose electrode selection to overlay")
    p.add_argument("--max-spikes-per-block", type=int, default=0,
                   help="Cap spikes read per block (0 = all); useful for a quick look")
    p.add_argument("--no-figures", action="store_true", help="Metrics only, no plots")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    targets = discover(a.path, a.assay_subfolder or None)
    if not targets:
        raise SystemExit(f"No {'data.raw.h5'} found under {a.path}")
    LOG.info("Processing %d recording(s)", len(targets))

    summaries = []
    for t in targets:
        try:
            s = process_file(t, a.output_dir, figures=not a.no_figures,
                             active_hz=a.active_hz, selection_from=a.selection_from,
                             max_spikes_per_block=a.max_spikes_per_block)
            if s:
                summaries.append(s)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            LOG.exception("Failed on %s: %s", t, exc)

    if summaries:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        index = a.output_dir / "activity_scan_index.json"
        index.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs": summaries,
        }, indent=2))
        total = sum(len(s["wells"]) for s in summaries)
        failing = sum(1 for s in summaries for w in s["wells"] if w["qc"] != "pass")
        LOG.info("Done — %d run(s), %d well(s), %d flagged. Index: %s",
                 len(summaries), total, failing, index)


if __name__ == "__main__":
    main()
