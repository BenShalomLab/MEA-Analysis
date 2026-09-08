# ==========================================================
# burst_common.py
# Author: Mandar Patil
# LLM Assisted Edits: Yes Claude Sonnet 5
#
# Shared helpers used by every network-burst detector
# (parameter_free_burst_detector.py, gaussianNetworkBursts.py, ...)
# so metrics stay identical across detectors and don't drift.
# ==========================================================

import numpy as np


def stats(x):
    """Mean/std/CV of a 1D array-like. Empty input -> zeros, not NaN,
    so downstream JSON/Excel consumers never see a stray null."""
    x = np.asarray(x)

    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}

    mean_val = x.mean()
    std_val = x.std()
    cv = std_val / mean_val if abs(mean_val) > 1e-12 else np.nan

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "cv": float(cv),
    }


def level_metrics(events, total_dur, ibi_key="ibi_s"):
    """Aggregate per-event dicts (as produced by any detector's event list)
    into the summary block used under burst_fragments/network_bursts/
    superbursts in network_results.json.

    `events` entries only need "start_time_s" and "burst_duration_s" plus
    whichever optional fields ("burst_area", "participation_fraction",
    "spike_count", "peak_population_firing_rate_hz",
    "peak_participation_fraction") are present; missing optional fields are
    simply omitted from the summary instead of raising.
    """
    if not events:
        return {}

    starts = [ev["start_time_s"] for ev in events]

    summary = {
        "burst_count": len(events),
        "burst_rate_hz": len(events) / total_dur if total_dur > 0 else 0.0,
        "burst_duration_s": stats([ev["burst_duration_s"] for ev in events]),
        ibi_key: stats(np.diff(starts)) if len(starts) > 1 else stats([]),
    }

    optional_fields = (
        "burst_area",
        "participation_fraction",
        "spike_count",
        "peak_population_firing_rate_hz",
        "peak_participation_fraction",
    )
    for field in optional_fields:
        if all(field in ev for ev in events):
            key = "spike_count_per_burst" if field == "spike_count" else field
            summary[key] = stats([ev[field] for ev in events])

    return summary
