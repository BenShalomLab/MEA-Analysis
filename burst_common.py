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


# Version of the network_results.json metric schema. Stamped into every result
# so a file can be read without guessing which code produced it.
#
#   1  pre-Epic-A. Rates divided by the spike span, not the recording;
#      stats([]) returned zeros rather than nulls; superbursts required only
#      one component; the Gaussian detector's rate was not divided by n_units
#      and its burst edges used the inverse of the intended fraction of peak.
#   2  Epic A. Those six defects corrected. Added burst_duration_p95_s /
#      _max_s and the offset-to-onset <ibi>_gap_s convention, plus the
#      detector / duration_source diagnostics.
#   3  Intensity made yield-normalised and internally consistent:
#      peak_population_firing_rate_hz is now per unit (the array-wide sum
#      moved to peak_population_firing_rate_total_hz); merged tiers
#      re-integrate over the whole burst window instead of summing
#      components; spikes_per_burst_per_unit, burst_density_hz,
#      peak_synchrony and peak_bin_synchrony are reported at every tier;
#      component_count renamed n_fragments.
#
# Units for every key are in docs/metrics.md and metrics.yaml.
SCHEMA_VERSION = 3


def stats(x):
    """Mean/std/CV of a 1D array-like.

    Empty input returns None (JSON null) for every field, NOT zeros. A well
    with no detected bursts has an *undefined* mean burst duration; encoding
    that as 0.0 silently drags down group means and inflates group variance
    in any downstream comparison (a silent knockout well would look like it
    had very short bursts rather than none). Consumers distinguish the two
    cases via `burst_count`, which is always a real number.

    `cv` is None when the mean is ~0, since the ratio is undefined there.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return {"mean": None, "std": None, "cv": None}

    mean_val = float(x.mean())
    std_val = float(x.std())
    cv = (std_val / mean_val) if abs(mean_val) > 1e-12 else None

    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
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

    `total_dur` must be the RECORDING duration in seconds, not the span
    between the first and last spike — otherwise a well that falls silent
    halfway through reports a burst rate computed over only its active
    period, which inflates it relative to a continuously active well.

    Two inter-burst interval conventions are reported because the
    literature is split and they diverge for long bursts:
      * `<ibi_key>`     onset-to-onset (burst cycle period)
      * `<ibi_key>_gap` offset-to-onset (silent gap; the Mossink et al. 2021
                        / Axion convention)
    """
    if not events:
        return {"burst_count": 0, "burst_rate_hz": 0.0}

    events_sorted = sorted(events, key=lambda ev: ev["start_time_s"])
    starts = np.asarray([ev["start_time_s"] for ev in events_sorted], dtype=float)
    ends = np.asarray([ev["end_time_s"] for ev in events_sorted], dtype=float)
    durations = np.asarray([ev["burst_duration_s"] for ev in events_sorted], dtype=float)

    # Offset-to-onset gaps; clipped at 0 so overlapping events (possible when
    # a detector emits nested tiers) do not contribute negative "silences".
    gaps = np.clip(starts[1:] - ends[:-1], 0.0, None) if len(starts) > 1 else np.asarray([])

    gap_key = f"{ibi_key[:-2]}_gap_s" if ibi_key.endswith("_s") else f"{ibi_key}_gap"

    summary = {
        "burst_count": len(events),
        "burst_rate_hz": len(events) / total_dur if total_dur > 0 else 0.0,
        "burst_duration_s": stats(durations),
        # Long single bursts are biologically distinct from clusters of short
        # ones; the mean alone hides that, so keep the upper tail explicitly.
        "burst_duration_p95_s": float(np.percentile(durations, 95)),
        "burst_duration_max_s": float(np.max(durations)),
        ibi_key: stats(starts[1:] - starts[:-1]) if len(starts) > 1 else stats([]),
        gap_key: stats(gaps),
    }

    optional_fields = (
        "burst_area",
        "participation_fraction",
        "spike_count",
        # Yield-normalised intensity. spike_count and
        # peak_population_firing_rate_total_hz both rise with the number of
        # sorted units, so these three are the ones to compare across wells.
        "spikes_per_burst_per_unit",
        "burst_density_hz",
        "peak_population_firing_rate_hz",
        "peak_population_firing_rate_total_hz",
        "peak_synchrony",
        "peak_participation_fraction",
        "peak_bin_synchrony",
    )
    for field in optional_fields:
        if all(field in ev for ev in events):
            key = "spike_count_per_burst" if field == "spike_count" else field
            summary[key] = stats([ev[field] for ev in events])

    return summary
