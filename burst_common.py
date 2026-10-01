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
#   3  Intensity made yield-normalised and internally consistent: the peak
#      firing rate became per unit (the array-wide sum moved to its own key);
#      merged tiers re-integrate over the whole burst window instead of
#      summing components; per-unit spike count, burst density and raw peak
#      synchrony reported at every tier; component_count renamed n_fragments.
#   4  Every key renamed to the published vocabulary, and the set completed.
#      Renames (old -> new):
#        peak_population_firing_rate_hz       -> burst_peak_hz_per_unit
#        peak_population_firing_rate_total_hz -> burst_peak_hz_array
#        peak_participation_fraction          -> coactive_fraction_peak
#        peak_bin_synchrony                   -> coactive_fraction_max
#        spike_count / spike_count_per_burst  -> spikes_per_burst
#        burst_area                           -> burst_area_spikes_per_unit
#        burst_density_hz                     -> intraburst_rate_hz
#        participation_fraction_signal        -> coactive_fraction_signal
#        participation_baseline/_mad/_bc      -> coactive_fraction_baseline
#                                                /_mad /_bimodality
#        cv_isi                               -> isi_cv
#        bimodality_coefficient               -> log_isi_bimodality
#        is_bursty                            -> is_bursty_by_isi_statistics
#      Added: rise_time_s and decay_time_s (RT, DT), intraburst_isi_mean_s
#      (bISI), burst_rate_per_min (the unit NBR is quoted in), duty_cycle,
#      and the spike_participation block carrying percent_random_spikes
#      (PRS). participation_fraction keeps its name: it is the published term
#      for breadth of recruitment.
#
#   5  Merging restructured so each tier's criterion comes from its own
#      level. Fragment -> network burst is now gated on continuity of the
#      population co-activity alone, with no gap: the previous gap came from
#      within-burst SPIKE intervals of single units (2-16 ms) while sub-burst
#      pauses are 80-500 ms, so that tier merged nothing and n_fragments was
#      always 1. Network burst -> superburst is now derived from the network
#      bursts' OWN inter-burst intervals, split by Otsu's method and gated on
#      their bimodality, rather than from inter-fragment intervals with a
#      hardcoded 0.75 s fallback; a culture whose intervals are unimodal is
#      reported as having no superbursts instead of manufactured ones.
#      Diagnostics renamed accordingly: nb_merge_gap_s -> superburst_gap_s,
#      nb_merge_gap_source -> superburst_gap_source; fragment_merge_gap_s and
#      superburst_merge_gap_s removed; fragment_merge_rule,
#      fragment_max_gap_s and nb_ibi_bimodality added.
#
# Units, formula and literature for every key are in metrics.json, rendered
# to docs/metrics.md.
SCHEMA_VERSION = 5


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


def intraburst_isi_mean(all_spikes_sorted, start_s, end_s):
    """Mean interval between consecutive spikes inside one burst window.

    Pooled across units, so it measures how densely the network fires during
    the burst rather than any one neuron's rhythm. "bISI" in MaxLab Live.
    None when fewer than two spikes fall inside the window.
    """
    spikes = np.asarray(all_spikes_sorted, dtype=float)
    left = np.searchsorted(spikes, start_s, side="left")
    right = np.searchsorted(spikes, end_s, side="right")
    if right - left < 2:
        return None
    return float(np.mean(np.diff(spikes[left:right])))


def spike_participation(all_spikes_sorted, events):
    """How much of the well's spiking happens inside network bursts.

    `percent_random_spikes` is the complement: the share of spikes firing
    outside any network burst. PRS in Mossink et al. 2021, one of their core
    parameters, because a culture can keep its burst rate while its neurons
    drift out of the bursts and only this ratio shows that.

    Overlapping event windows are merged first, so a spike is never counted
    twice.
    """
    spikes = np.asarray(all_spikes_sorted, dtype=float)
    n_total = int(spikes.size)
    result = {
        "n_spikes_total": n_total,
        "n_spikes_in_network_bursts": 0,
        "fraction_spikes_in_network_bursts": 0.0 if n_total else None,
        "percent_random_spikes": 100.0 if n_total else None,
    }
    if n_total == 0 or not events:
        return result

    windows = sorted((ev["start_time_s"], ev["end_time_s"]) for ev in events)
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    n_inside = 0
    for start, end in merged:
        left = np.searchsorted(spikes, start, side="left")
        right = np.searchsorted(spikes, end, side="right")
        n_inside += int(right - left)

    result["n_spikes_in_network_bursts"] = n_inside
    result["fraction_spikes_in_network_bursts"] = n_inside / n_total
    result["percent_random_spikes"] = 100.0 * (1.0 - n_inside / n_total)
    return result


def level_metrics(events, total_dur, ibi_key="ibi_s"):
    """Aggregate per-event dicts (as produced by any detector's event list)
    into the summary block used under burst_fragments/network_bursts/
    superbursts in network_results.json.

    `events` entries only need "start_time_s" and "burst_duration_s" plus
    whichever optional fields ("burst_area_spikes_per_unit", "participation_fraction",
    "spikes_per_burst", "burst_peak_hz_per_unit",
    "coactive_fraction_peak") are present; missing optional fields are
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
        return {"burst_count": 0, "burst_rate_hz": 0.0, "burst_rate_per_min": 0.0}

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
        # The same rate in the unit the literature quotes it in: Mossink et
        # al. 2021 define NBR per minute, and the well quality gates in the
        # field (>0.4 bursts/min, >1 network burst/min) are stated that way.
        # Reported explicitly so nobody compares a Hz value to a per-minute
        # threshold.
        "burst_rate_per_min": (len(events) / total_dur * 60.0) if total_dur > 0 else 0.0,
        "burst_duration_s": stats(durations),
        # Long single bursts are biologically distinct from clusters of short
        # ones; the mean alone hides that, so keep the upper tail explicitly.
        "burst_duration_p95_s": float(np.percentile(durations, 95)),
        "burst_duration_max_s": float(np.max(durations)),
        ibi_key: stats(starts[1:] - starts[:-1]) if len(starts) > 1 else stats([]),
        gap_key: stats(gaps),
    }

    mean_duration = float(durations.mean())
    mean_gap = float(gaps.mean()) if gaps.size else None
    if mean_gap is not None and (mean_duration + mean_gap) > 0:
        # Fraction of time the network spends inside a burst. The
        # parameter-free factor of the effective excitability in Vinogradov et
        # al. 2024 (their alpha is this times a model scale constant),
        # reported on its own so it can be compared without committing to
        # their model fit. It rises both when bursts lengthen and when they
        # come closer together.
        summary["duty_cycle"] = mean_duration / (mean_duration + mean_gap)

    optional_fields = (
        # Shape. Mossink et al. 2021 report rise and decay separately because
        # they dissociate in disease models where duration and rate do not.
        "rise_time_s",
        "decay_time_s",
        # Breadth and simultaneity: different measurements, not two
        # aggregations of one.
        "participation_fraction",
        "coactive_fraction_peak",
        "coactive_fraction_max",
        # Intensity. spikes_per_burst and burst_peak_hz_array rise with the
        # number of sorted units; the other three do not, so those are the
        # ones to compare across wells.
        "spikes_per_burst",
        "spikes_per_burst_per_unit",
        "burst_area_spikes_per_unit",
        "intraburst_rate_hz",
        "intraburst_isi_mean_s",
        "burst_peak_hz_per_unit",
        "burst_peak_hz_array",
    )
    for field in optional_fields:
        values = [ev[field] for ev in events if ev.get(field) is not None]
        if values and all(field in ev for ev in events):
            summary[field] = stats(values)

    return summary
