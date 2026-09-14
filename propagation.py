# ==========================================================
# propagation.py
#
# Where network bursts start and how they spread.
#
# A network burst is not instantaneous: units join it in a repeatable order,
# some consistently leading and others following. That order is a property of
# the circuit rather than of any single unit's firing rate, and it changes in
# disease models where burst rate and duration do not. It is only measurable
# with a high-density array, because it needs sorted units with known
# positions rather than a handful of electrodes.
#
# Two things are reported:
#   * per unit, how early it joins bursts (a leader score) and how reliably
#   * per burst, how fast activity travels outward from its origin, by
#     regressing each unit's distance from the origin on its latency
# ==========================================================

from __future__ import annotations

import numpy as np

# A burst needs this many participating units before its timing is worth
# fitting; below it the latency spread is dominated by which units happened
# to fire.
DEFAULT_MIN_UNITS_PER_BURST = 5

# Minimum fraction of variance the distance-latency fit must explain before
# its slope is reported as a propagation speed.
DEFAULT_MIN_R_SQUARED = 0.1


def _first_spike_in_window(times, start, end):
    """First spike at or after `start` and at or before `end`, else None."""
    index = np.searchsorted(times, start, side="left")
    if index >= times.size or times[index] > end:
        return None
    return float(times[index])


def burst_latencies(SpikeTimes, events, min_units_per_burst=DEFAULT_MIN_UNITS_PER_BURST):
    """Per-burst participation times, relative to the first unit to fire.

    Aligning to the earliest participating unit rather than to the detector's
    burst start keeps the latencies independent of how the detector drew the
    burst boundary, so they stay comparable between detectors and between
    thresholds.

    Returns a list of {unit_id: latency_s} dicts, one per analysed burst.
    """
    unit_ids = list(SpikeTimes.keys())
    trains = {u: np.sort(np.asarray(SpikeTimes[u], dtype=float)) for u in unit_ids}

    per_burst = []
    for event in events:
        start, end = event["start_time_s"], event["end_time_s"]
        first_spikes = {}
        for unit_id in unit_ids:
            spike = _first_spike_in_window(trains[unit_id], start, end)
            if spike is not None:
                first_spikes[unit_id] = spike

        if len(first_spikes) < min_units_per_burst:
            continue

        origin_time = min(first_spikes.values())
        per_burst.append({u: t - origin_time for u, t in first_spikes.items()})

    return per_burst


def _unit_summary(per_burst, unit_ids, n_bursts_analysed):
    """Per-unit latency statistics and leader score."""
    latencies = {u: [] for u in unit_ids}
    ranks = {u: [] for u in unit_ids}

    for burst in per_burst:
        order = sorted(burst, key=burst.get)
        n_participants = len(order)
        for position, unit_id in enumerate(order):
            latencies[unit_id].append(burst[unit_id])
            # Normalised rank: 0 for the first unit, 1 for the last, so the
            # score does not depend on how many units joined this burst.
            ranks[unit_id].append(position / (n_participants - 1) if n_participants > 1 else 0.0)

    summary = {}
    for unit_id in unit_ids:
        values = np.asarray(latencies[unit_id], dtype=float)
        if values.size == 0:
            summary[unit_id] = {
                "prop_n_bursts": 0,
                "prop_participation_fraction": 0.0,
                "prop_mean_latency_ms": None,
                "prop_latency_sd_ms": None,
                "prop_leader_score": None,
            }
            continue
        summary[unit_id] = {
            "prop_n_bursts": int(values.size),
            "prop_participation_fraction": (
                float(values.size / n_bursts_analysed) if n_bursts_analysed else None
            ),
            "prop_mean_latency_ms": float(values.mean() * 1000.0),
            "prop_latency_sd_ms": float(values.std() * 1000.0) if values.size > 1 else None,
            # 1 means the unit is always first to fire, 0 always last.
            "prop_leader_score": float(1.0 - np.mean(ranks[unit_id])),
        }
    return summary


def _propagation_speed(burst, locations, min_r_squared):
    """Speed in um/ms from a distance-on-latency fit, or None.

    The origin is the location of the first unit to fire. Distance is fitted
    as a function of latency, so the slope is a speed; a poor fit means the
    burst did not spread as a wave and no speed is reported for it.
    """
    usable = [(u, lat) for u, lat in burst.items() if u in locations]
    if len(usable) < 3:
        return None

    usable.sort(key=lambda item: item[1])
    origin = np.asarray(locations[usable[0][0]], dtype=float)

    latencies = np.asarray([lat for _, lat in usable], dtype=float)
    distances = np.asarray(
        [np.linalg.norm(np.asarray(locations[u], dtype=float) - origin) for u, _ in usable]
    )

    if np.ptp(latencies) <= 0 or np.ptp(distances) <= 0:
        return None

    slope, intercept = np.polyfit(latencies, distances, 1)
    predicted = slope * latencies + intercept
    residual = float(np.sum((distances - predicted) ** 2))
    total = float(np.sum((distances - distances.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0

    if r_squared < min_r_squared or slope <= 0:
        return {"speed_um_per_ms": None, "r_squared": float(r_squared)}

    # slope is um per second; report um per millisecond.
    return {"speed_um_per_ms": float(slope / 1000.0), "r_squared": float(r_squared)}


def compute_propagation(SpikeTimes, events, unit_locations=None,
                        min_units_per_burst=DEFAULT_MIN_UNITS_PER_BURST,
                        min_r_squared=DEFAULT_MIN_R_SQUARED):
    """Burst propagation for one well.

    `unit_locations` maps unit id to (x, y) in microns; without it the
    per-unit leader scores are still computed and only the speeds are skipped.

    Returns {"units": {...}, "summary": {...}}.
    """
    unit_ids = list(SpikeTimes.keys())
    if not events or not unit_ids:
        return {"units": {}, "summary": {"n_bursts_analysed": 0}}

    per_burst = burst_latencies(SpikeTimes, events, min_units_per_burst)
    if not per_burst:
        return {
            "units": {},
            "summary": {
                "n_bursts_analysed": 0,
                "n_bursts_total": len(events),
                "reason": "no_burst_had_enough_participating_units",
            },
        }

    units = _unit_summary(per_burst, unit_ids, len(per_burst))

    spreads = np.asarray(
        [max(b.values()) - min(b.values()) for b in per_burst], dtype=float
    )
    summary = {
        "n_bursts_analysed": len(per_burst),
        "n_bursts_total": len(events),
        "mean_participants_per_burst": float(np.mean([len(b) for b in per_burst])),
        "latency_spread_mean_ms": float(spreads.mean() * 1000.0),
        "latency_spread_cv": (
            float(spreads.std() / spreads.mean()) if spreads.mean() > 0 else None
        ),
    }

    # How repeatable the firing order is. A network with consistent leaders
    # has a wide spread of leader scores; one where order is random has all
    # units near 0.5.
    leader_scores = np.asarray(
        [u["prop_leader_score"] for u in units.values() if u["prop_leader_score"] is not None],
        dtype=float,
    )
    if leader_scores.size:
        summary["leader_score_std"] = float(leader_scores.std())
        summary["n_units_participating"] = int(leader_scores.size)

    if unit_locations:
        fits = [
            _propagation_speed(burst, unit_locations, min_r_squared)
            for burst in per_burst
        ]
        fits = [f for f in fits if f is not None]
        speeds = np.asarray(
            [f["speed_um_per_ms"] for f in fits if f["speed_um_per_ms"] is not None],
            dtype=float,
        )
        r_squared = np.asarray([f["r_squared"] for f in fits], dtype=float)
        summary["n_bursts_with_speed_fit"] = int(speeds.size)
        summary["fraction_bursts_with_speed_fit"] = (
            float(speeds.size / len(per_burst)) if per_burst else None
        )
        summary["median_fit_r_squared"] = (
            float(np.median(r_squared)) if r_squared.size else None
        )
        if speeds.size:
            summary["median_speed_um_per_ms"] = float(np.median(speeds))
            summary["speed_iqr_um_per_ms"] = float(
                np.percentile(speeds, 75) - np.percentile(speeds, 25)
            )

    return {"units": units, "summary": summary}
