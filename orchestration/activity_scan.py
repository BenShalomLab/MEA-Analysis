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

    # Per-block derived results, aggregated at the end. Kept per block because
    # electrodes in different blocks were never recorded simultaneously, so
    # anything relational (correlation, synchrony) is only valid within a block.
    temporal_blocks: list[dict] = field(default_factory=list)
    network_blocks: list[dict] = field(default_factory=list)
    functional_blocks: list[dict] = field(default_factory=list)
    # Population trace of the longest block, for plotting.
    pop_trace: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    pop_bursts: list[tuple[float, float]] = field(default_factory=list)
    raster_sample: Optional[tuple[np.ndarray, np.ndarray]] = None

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
# Temporal / network / functional analysis
#
# All of this comes from the spike timestamps (``frameno`` at the sampling rate)
# that the counting pass already reads, so it costs no extra I/O.
# --------------------------------------------------------------------------- #
def isi_metrics(times: np.ndarray, burst_isi_s: float = 0.1) -> tuple[float, float]:
    """Return ``(isi_cv, burst_fraction)`` for one electrode's spike times.

    ``isi_cv`` — coefficient of variation of inter-spike intervals. Near 0 is
    metronomic, ~1 is Poisson, well above 1 means bursty (long silences broken
    by rapid volleys).

    ``burst_fraction`` — share of intervals shorter than ``burst_isi_s``, i.e.
    how much of the firing happens inside high-frequency runs.
    """
    if times.size < 5:
        return 0.0, 0.0
    isi = np.diff(np.sort(times))
    isi = isi[isi > 0]
    if isi.size < 4:
        return 0.0, 0.0
    mean = float(isi.mean())
    cv = float(isi.std() / mean) if mean > 0 else 0.0
    return cv, float((isi < burst_isi_s).mean())


def rate_stability(times: np.ndarray, duration_s: float, window_s: float = 10.0) -> float:
    """CV of firing rate across time windows — detects drift or dying cultures.

    0 means perfectly steady; large values mean the rate changed a lot over the
    recording, which usually indicates a problem rather than biology.
    """
    if times.size < 10 or duration_s < 2 * window_s:
        return 0.0
    edges = np.arange(0, duration_s + window_s, window_s)
    counts, _ = np.histogram(times, bins=edges)
    if counts.mean() <= 0:
        return 0.0
    return float(counts.std() / counts.mean())


def detect_network_bursts(pop: np.ndarray, bin_s: float,
                          n_sigma: float = 4.0, min_gap_s: float = 0.2
                          ) -> tuple[list[tuple[float, float]], dict]:
    """Detect network bursts in a binned population spike count.

    Threshold is median + ``n_sigma`` x a robust spread (MAD-based), so a few
    huge bursts do not raise the threshold above the rest. Bins above threshold
    that are within ``min_gap_s`` of each other are merged into one burst.

    This is intentionally simple and assumption-light — it is an early preview
    from unsorted threshold crossings, not a replacement for the pipeline's
    parameter-free burst detector, which works on sorted units.
    """
    if pop.size < 10 or pop.sum() == 0:
        return [], {}

    med = float(np.median(pop))
    mad = float(np.median(np.abs(pop - med))) * 1.4826
    spread = mad if mad > 0 else float(pop.std())
    if spread <= 0:
        return [], {}
    threshold = med + n_sigma * spread

    above = pop > threshold
    if not above.any():
        return [], {"threshold": round(threshold, 2)}

    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(above.size)

    merged: list[tuple[int, int]] = []
    for s, e in zip(starts, stops):
        if merged and (s - merged[-1][1]) * bin_s <= min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    bursts = [(s * bin_s, e * bin_s) for s, e in merged if (e - s) * bin_s > 0]
    if not bursts:
        return [], {"threshold": round(threshold, 2)}

    durations = np.array([e - s for s, e in bursts])
    spikes_in = np.array([pop[int(s / bin_s):int(e / bin_s)].sum() for s, e in bursts])
    total_s = pop.size * bin_s
    ibis = np.diff([s for s, _ in bursts]) if len(bursts) > 1 else np.array([])

    stats = {
        "threshold": round(threshold, 2),
        "network_burst_count": len(bursts),
        "network_burst_rate_hz": round(len(bursts) / total_s, 4) if total_s else 0.0,
        "burst_duration_mean_s": round(float(durations.mean()), 3),
        "burst_duration_median_s": round(float(np.median(durations)), 3),
        "spikes_per_burst_mean": round(float(spikes_in.mean()), 1),
        "pct_spikes_in_bursts": round(100 * float(spikes_in.sum()) / float(pop.sum()), 1),
        "interburst_interval_mean_s": round(float(ibis.mean()), 3) if ibis.size else None,
    }
    return bursts, stats


def population_metrics(times: np.ndarray, duration_s: float,
                       bin_s: float = 0.02) -> tuple[np.ndarray, np.ndarray, dict]:
    """Binned population activity plus synchrony measures.

    ``fano`` (variance/mean of the binned count) is the headline synchrony
    number: ~1 means independent firing, large values mean the population rises
    and falls together.
    """
    if times.size == 0 or duration_s <= 0:
        return np.empty(0), np.empty(0), {}
    edges = np.arange(0, duration_s + bin_s, bin_s)
    pop, _ = np.histogram(times, bins=edges)
    centres = edges[:-1] + bin_s / 2
    mean = float(pop.mean())
    stats = {
        "population_rate_hz": round(float(times.size / duration_s), 2),
        "pop_bin_mean": round(mean, 2),
        "pop_bin_max": int(pop.max()),
        "synchrony_fano": round(float(pop.var() / mean), 2) if mean > 0 else 0.0,
        "pop_bin_s": bin_s,
    }
    return centres, pop, stats


def functional_metrics(times: np.ndarray, channels: np.ndarray,
                       chan_to_elec: dict[int, int], duration_s: float,
                       max_electrodes: int = 400, bin_s: float = 0.01
                       ) -> tuple[dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """Pairwise co-activation among simultaneously recorded electrodes.

    Only electrodes within the *same* block are comparable: a scan records
    different configurations at different times, so electrodes from different
    blocks never overlap in time and cannot be correlated.

    Returns ``(stats, corr_matrix, electrode_ids)``; the matrix is capped at
    ``max_electrodes`` busiest electrodes to keep this affordable.
    """
    if times.size == 0 or duration_s <= 0:
        return {}, None, None

    counts = np.bincount(channels)
    busiest = np.argsort(counts)[::-1]
    busiest = [c for c in busiest if counts[c] > 10][:max_electrodes]
    if len(busiest) < 10:
        return {}, None, None

    n_bins = max(int(duration_s / bin_s), 2)
    mat = np.zeros((len(busiest), n_bins), dtype=np.float32)
    for row, ch in enumerate(busiest):
        t = times[channels == ch]
        idx = np.clip((t / bin_s).astype(np.int64), 0, n_bins - 1)
        np.add.at(mat[row], idx, 1.0)

    keep = mat.std(axis=1) > 0
    mat, busiest = mat[keep], [c for c, k in zip(busiest, keep) if k]
    if mat.shape[0] < 10:
        return {}, None, None

    corr = np.corrcoef(mat)
    np.fill_diagonal(corr, np.nan)
    off = corr[~np.isnan(corr)]
    if off.size == 0:
        return {}, None, None

    # "Connected" pairs: correlation clearly above the bulk of the distribution.
    strong = float(np.percentile(off, 95))
    degree = np.nansum(corr > strong, axis=1)

    stats = {
        "functional_electrodes": int(mat.shape[0]),
        "correlation_mean": round(float(off.mean()), 4),
        "correlation_median": round(float(np.median(off)), 4),
        "correlation_p95": round(strong, 4),
        "correlation_max": round(float(off.max()), 4),
        "mean_degree": round(float(degree.mean()), 1),
        "corr_bin_s": bin_s,
    }
    elec_ids = np.array([chan_to_elec.get(int(c), -1) for c in busiest], dtype=np.int64)
    return stats, corr, elec_ids


def _weighted_mean(blocks: list[dict], key: str, weight: str = "duration") -> Optional[float]:
    vals = [(b[key], b.get(weight, 1.0)) for b in blocks
            if b.get(key) is not None and not isinstance(b.get(key), str)]
    if not vals:
        return None
    total_w = sum(w for _, w in vals) or 1.0
    return float(sum(v * w for v, w in vals) / total_w)


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


def extract_file(path: Path, max_spikes_per_block: int = 0, *,
                 analyze: bool = True, functional: bool = True,
                 max_corr_electrodes: int = 400) -> dict[int, WellActivity]:
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
                frame = data["frameno"].astype(np.int64)

                valid = (sch >= 0) & (sch < n_chan)
                sch, amp, frame = sch[valid], amp[valid], frame[valid]
                idx = chan_to_idx[sch]
                ok = idx >= 0
                idx, amp, frame, sch = idx[ok], amp[ok], frame[ok], sch[ok]

                counts = np.bincount(idx, minlength=elec.size)
                amps = np.bincount(idx, weights=amp, minlength=elec.size)
                hit = np.nonzero(counts)[0]
                for i in hit:
                    rec = store[int(elec[i])]
                    rec[0] += float(counts[i])
                    rec[2] += float(amps[i])

                # --- timing-based analysis (free: the data is already loaded) ---
                if analyze and frame.size:
                    fs = wa.sampling_hz or 10000.0
                    t = (frame - frame.min()) / fs
                    span = float(t.max()) if t.size else 0.0
                    dur = duration if duration > 0 else span
                    order = np.argsort(t)
                    t, sch_o, idx_o = t[order], sch[order], idx[order]

                    # Per-electrode temporal character.
                    cvs, bfs, stabs = [], [], []
                    for i in hit:
                        if counts[i] < 5:
                            continue
                        te = t[idx_o == i]
                        cv, bf = isi_metrics(te)
                        if cv:
                            cvs.append(cv)
                            bfs.append(bf)
                            stabs.append(rate_stability(te, dur))
                    wa.temporal_blocks.append({
                        "duration": dur,
                        "isi_cv_median": float(np.median(cvs)) if cvs else None,
                        "burst_fraction_mean": float(np.mean(bfs)) if bfs else None,
                        "rate_stability_cv": float(np.median(stabs)) if stabs else None,
                        "electrodes": len(cvs),
                    })

                    # Population activity and network bursts for this block.
                    centres, pop, pstats = population_metrics(t, dur)
                    if pstats:
                        bursts, bstats = detect_network_bursts(pop, pstats["pop_bin_s"])
                        pstats.update(bstats)
                        pstats["duration"] = dur
                        wa.network_blocks.append(pstats)
                        # Keep the longest block's trace for the figure.
                        if wa.pop_trace is None or dur > wa.pop_trace[2]:
                            wa.pop_trace = (centres, pop, dur)
                            wa.pop_bursts = bursts
                            step = max(1, t.size // 60000)   # cap raster points
                            wa.raster_sample = (t[::step], idx_o[::step].astype(np.float64))

                    # Co-activation among electrodes recorded together here.
                    if functional:
                        c2e = {int(chan[i]): int(elec[i]) for i in range(elec.size)}
                        fstats, corr, eids = functional_metrics(
                            t, sch_o, c2e, dur,
                            max_electrodes=max_corr_electrodes)
                        if fstats:
                            fstats["duration"] = dur
                            wa.functional_blocks.append(fstats)
                            if corr is not None and (
                                    not hasattr(wa, "_corr") or wa._corr is None
                                    or corr.shape[0] > wa._corr.shape[0]):
                                wa._corr = corr           # type: ignore[attr-defined]
                                wa._corr_elec = eids      # type: ignore[attr-defined]

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

    # --- temporal character ------------------------------------------------
    if wa.temporal_blocks:
        for key, out in (("isi_cv_median", "isi_cv_median"),
                         ("burst_fraction_mean", "burst_fraction_mean"),
                         ("rate_stability_cv", "rate_stability_cv")):
            v = _weighted_mean(wa.temporal_blocks, key)
            if v is not None:
                m[out] = round(v, 4)

    # --- population / network bursts ---------------------------------------
    if wa.network_blocks:
        for key in ("population_rate_hz", "synchrony_fano", "network_burst_rate_hz",
                    "burst_duration_mean_s", "spikes_per_burst_mean",
                    "pct_spikes_in_bursts", "interburst_interval_mean_s"):
            v = _weighted_mean(wa.network_blocks, key)
            if v is not None:
                m[key] = round(v, 4)
        m["network_burst_count"] = int(sum(b.get("network_burst_count", 0)
                                           for b in wa.network_blocks))

    # --- functional connectivity -------------------------------------------
    if wa.functional_blocks:
        for key in ("correlation_mean", "correlation_median", "correlation_p95",
                    "correlation_max", "mean_degree"):
            v = _weighted_mean(wa.functional_blocks, key)
            if v is not None:
                m[key] = round(v, 4)
        m["functional_electrodes"] = int(max(
            b.get("functional_electrodes", 0) for b in wa.functional_blocks))

    # --- how well the recorded electrodes captured the available activity ---
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

            total_spikes = float(wa.spikes.sum())
            if total_spikes > 0:
                m["captured_activity_fraction"] = round(
                    float(wa.spikes[sel_mask].sum() / total_spikes), 4)

            # Recall against an ideal "take the n busiest electrodes" selection:
            # 1.0 means the selection found everything it could have.
            order = np.argsort(rate)[::-1][:n_sel]
            ideal = set(wa.electrode[order].tolist())
            chosen = set(wa.electrode[sel_mask].tolist())
            m["selection_recall"] = round(len(ideal & chosen) / max(len(ideal), 1), 3)

            ideal_rate = float(rate[order].mean()) if order.size else 0.0
            m["selection_efficiency"] = round(
                float(sel_rate.mean() / ideal_rate), 3) if ideal_rate > 0 else None

            if m.get("selection_recall") is not None:
                eff = m.get("selection_efficiency") or 0
                m["selection_quality"] = (
                    "good" if eff >= 0.8 else "fair" if eff >= 0.5 else "poor")
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


def plot_network(wa: WellActivity, metrics: dict, out: Path) -> Optional[Path]:
    """Raster and population rate for the longest block, with bursts marked."""
    if wa.pop_trace is None:
        return None
    plt = _setup_mpl()
    centres, pop, dur = wa.pop_trace

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})

    ax = axes[0]
    if wa.raster_sample is not None:
        t, rows = wa.raster_sample
        ax.scatter(t, rows, s=0.09, c="#1d2939", linewidths=0, alpha=0.55)
    for s, e in wa.pop_bursts:
        ax.axvspan(s, e, color="#eb6834", alpha=0.16, linewidth=0)
    ax.set_ylabel("electrode (block index)")
    ax.set_title(f"Well {wa.meta.label or wa.well_id}"
                 + (f" · {wa.meta.group}" if wa.meta.group else "")
                 + "  —  spike raster with detected network bursts")

    ax = axes[1]
    ax.fill_between(centres, pop, color="#4f46e5", alpha=0.75, linewidth=0)
    for s, e in wa.pop_bursts:
        ax.axvspan(s, e, color="#eb6834", alpha=0.16, linewidth=0)
    ax.set_xlabel("time (s)")
    bin_ms = round(1000 * float(centres[1] - centres[0])) if centres.size > 1 else 20
    ax.set_ylabel(f"spikes / {bin_ms} ms")
    ax.set_xlim(0, dur)

    bits = [f"{metrics.get('network_burst_count', 0)} bursts"]
    if metrics.get("network_burst_rate_hz") is not None:
        bits.append(f"{metrics['network_burst_rate_hz']:.3f} Hz")
    if metrics.get("pct_spikes_in_bursts") is not None:
        bits.append(f"{metrics['pct_spikes_in_bursts']:.0f}% of spikes in bursts")
    if metrics.get("synchrony_fano") is not None:
        bits.append(f"Fano {metrics['synchrony_fano']:.1f}")
    fig.suptitle("   ·   ".join(bits), fontsize=9, y=0.98)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_functional(wa: WellActivity, metrics: dict, out: Path) -> Optional[Path]:
    """Co-activation matrix and the spatial layout of highly connected sites."""
    corr = getattr(wa, "_corr", None)
    eids = getattr(wa, "_corr_elec", None)
    if corr is None or eids is None:
        return None
    plt = _setup_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    finite = corr[~np.isnan(corr)]
    lim = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    im = ax.imshow(np.nan_to_num(corr), cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="correlation", fraction=0.046, pad=0.02)
    ax.set_title(f"Co-activation ({corr.shape[0]} busiest electrodes)")
    ax.set_xlabel("electrode"); ax.set_ylabel("electrode")

    ax = axes[1]
    strong = metrics.get("correlation_p95", 0) or 0
    degree = np.nansum(corr > strong, axis=1)
    pos = {int(e): i for i, e in enumerate(wa.electrode.tolist())}
    xs, ys, dd = [], [], []
    for e, d in zip(eids.tolist(), degree.tolist()):
        i = pos.get(int(e))
        if i is not None:
            xs.append(wa.x[i]); ys.append(wa.y[i]); dd.append(d)
    if xs:
        sc = ax.scatter(xs, ys, c=dd, s=12, cmap="viridis", linewidths=0)
        fig.colorbar(sc, ax=ax, label="co-active partners", fraction=0.046, pad=0.02)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title("Connectivity hubs")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

    fig.suptitle(
        f"mean r {metrics.get('correlation_mean', 0):.3f}   ·   "
        f"mean degree {metrics.get('mean_degree', 0)}   ·   "
        "within simultaneously-recorded electrodes only", fontsize=9, y=1.02)
    fig.tight_layout()
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
                 max_spikes_per_block: int = 0, analyze: bool = True,
                 functional: bool = True, max_corr_electrodes: int = 400) -> dict[str, Any]:
    """Extract, measure, plot, and write everything for a single recording."""
    LOG.info("Reading %s", h5_path)
    wells = extract_file(h5_path, max_spikes_per_block=max_spikes_per_block,
                         analyze=analyze, functional=functional,
                         max_corr_electrodes=max_corr_electrodes)
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
        extra = ""
        if m.get("network_burst_count") is not None:
            extra += f", {m['network_burst_count']} net bursts"
        if m.get("synchrony_fano") is not None:
            extra += f", Fano {m['synchrony_fano']:.1f}"
        if m.get("correlation_mean") is not None:
            extra += f", r {m['correlation_mean']:.3f}"
        LOG.info("  well %s (%s): %s active / %s scanned (%.1f%%), mean %.3f Hz%s — %s",
                 wa.meta.label or wid, wa.meta.group or "?",
                 f"{m['electrodes_active']:,}", f"{m['electrodes_scanned']:,}",
                 100 * m["active_fraction"], m.get("rate_mean_hz", 0), extra, verdict)

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
            plot_network(wa, metrics[wid], dest / f"well{wid:03d}_network.png")
            plot_functional(wa, metrics[wid], dest / f"well{wid:03d}_functional.png")
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
    p.add_argument("--no-temporal", action="store_true",
                   help="Skip timing analysis (ISI, network bursts, synchrony)")
    p.add_argument("--no-functional", action="store_true",
                   help="Skip pairwise co-activation analysis (the slowest part)")
    p.add_argument("--max-corr-electrodes", type=int, default=400,
                   help="Cap electrodes in the co-activation matrix (default 400)")
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
                             max_spikes_per_block=a.max_spikes_per_block,
                             analyze=not a.no_temporal,
                             functional=not (a.no_functional or a.no_temporal),
                             max_corr_electrodes=a.max_corr_electrodes)
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
