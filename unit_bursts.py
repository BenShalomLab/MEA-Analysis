# ==========================================================
# unit_bursts.py
#
# Single-unit burst detection and per-unit burst metrics.
#
# The pipeline measured bursting only at the network level, but most of the
# published phenotyping feature set is per unit: burst rate, burst duration,
# spikes per burst, intra-burst firing rate, fraction of spikes in bursts and
# the regularity of the intervals between bursts (Mossink et al. 2021 and the
# studies built on it). A network can look unchanged while its individual
# neurons burst quite differently.
#
# Two detectors are implemented, both recommended by the systematic comparison
# in Cotterill et al. 2016 (J Neurophysiol 116:306-321), which also advises
# running more than one method and comparing:
#
#   MaxInterval  fixed ISI thresholds. Cotterill's first choice: it behaves
#                predictably across developmental ages because its parameters
#                do not move with the data.
#   logISI       Pasquale et al. 2010. Picks the intra-burst ISI threshold per
#                unit from the antimode of its log-ISI distribution, so it
#                adapts to a unit's own firing statistics, and falls back to a
#                fixed cutoff when that distribution is not clearly bimodal.
#
# Both return the same metric names so they can be compared directly, and
# `agreement_jaccard` reports how far they agree on which spikes are in bursts.
# ==========================================================

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

METHODS = ("maxinterval", "logisi")

# MaxInterval defaults. These are the common MEA culture values (a 100 ms
# intra-burst interval and at least 5 spikes, as in Chiappalone et al. 2005),
# not values fitted to this lab's data. Cotterill et al. 2016 are explicit that
# MaxInterval performs best when its parameters are tuned to the dataset, so
# treat these as a starting point and record whatever you use.
MAXINTERVAL_DEFAULTS = {
    "max_isi_start_s": 0.100,   # ISI that may open a burst
    "max_isi_end_s": 0.200,     # ISI that may still extend an open burst
    "min_ibi_s": 0.200,         # bursts closer than this are merged
    "min_duration_s": 0.010,
    "min_spikes": 5,
}

# logISI defaults. The 100 ms cutoff is the intra-burst ISI ceiling used by
# Pasquale et al. 2010; the void parameter of 0.7 is their threshold for
# accepting two peaks in the log-ISI histogram as genuinely separate.
LOGISI_DEFAULTS = {
    "cutoff_s": 0.100,
    "min_void": 0.7,
    "min_spikes": 5,
    "n_bins": 100,
    "smoothing_bins": 2.0,
}


# ---------------------------------------------------------------------------
# Detectors: spike times -> list of (first index, last index) inclusive
# ---------------------------------------------------------------------------

def _merge_and_filter(bursts, times, min_ibi_s, min_duration_s, min_spikes):
    """Merge bursts separated by less than min_ibi_s, then drop the runts."""
    if not bursts:
        return []

    merged = [list(bursts[0])]
    for start, end in bursts[1:]:
        if times[start] - times[merged[-1][1]] < min_ibi_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
        if (end - start + 1) >= min_spikes
        and (times[end] - times[start]) >= min_duration_s
    ]


def maxinterval_bursts(times, **params):
    """MaxInterval burst detection. Returns inclusive (start, end) index pairs.

    A burst opens on an ISI at or below max_isi_start_s and continues while
    ISIs stay at or below the more permissive max_isi_end_s, which lets a burst
    decelerate towards its end without being cut in two.
    """
    options = {**MAXINTERVAL_DEFAULTS, **params}
    times = np.asarray(times, dtype=float)
    if times.size < options["min_spikes"]:
        return []

    times = np.sort(times)
    isis = np.diff(times)

    bursts = []
    index = 0
    n_isis = len(isis)
    while index < n_isis:
        if isis[index] > options["max_isi_start_s"]:
            index += 1
            continue
        start = index
        end = index + 1
        while end <= n_isis and isis[end - 1] <= options["max_isi_end_s"]:
            end += 1
        # The loop stops one past the last in-burst spike.
        bursts.append((start, end - 1))
        index = end

    return _merge_and_filter(
        bursts, times,
        options["min_ibi_s"], options["min_duration_s"], options["min_spikes"],
    )


def logisi_threshold(times, **params):
    """Per-unit intra-burst ISI threshold from the log-ISI distribution.

    Returns (threshold_s, source). `source` is "antimode" when two sufficiently
    separated peaks were found, "cutoff" when the distribution gave no usable
    split and the fixed ceiling was used instead. Reporting the source matters:
    a dataset where most units fall back to the cutoff is one where logISI is
    really just MaxInterval with one threshold.
    """
    options = {**LOGISI_DEFAULTS, **params}
    times = np.sort(np.asarray(times, dtype=float))
    isis = np.diff(times)
    isis = isis[isis > 0]
    if isis.size < 10:
        return options["cutoff_s"], "cutoff"

    log_isis = np.log10(isis)
    counts, edges = np.histogram(log_isis, bins=options["n_bins"])
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = gaussian_filter1d(counts.astype(float), sigma=options["smoothing_bins"])

    # find_peaks ignores maxima sitting on the first or last bin, and for a
    # cleanly bursting unit the intra-burst mode often is the first bin. Pad
    # with zeros so edge modes are detectable, then shift the indices back.
    padded = np.concatenate(([0.0], smoothed, [0.0]))
    peaks, _ = find_peaks(padded)
    peaks = np.clip(peaks - 1, 0, smoothed.size - 1)
    peaks = np.unique(peaks)
    if peaks.size < 2:
        return options["cutoff_s"], "cutoff"

    log_cutoff = np.log10(options["cutoff_s"])
    intra_candidates = peaks[centers[peaks] <= log_cutoff]
    if intra_candidates.size == 0:
        return options["cutoff_s"], "cutoff"

    # Tallest peak below the cutoff is the intra-burst mode.
    intra_peak = intra_candidates[np.argmax(smoothed[intra_candidates])]

    for later_peak in peaks[peaks > intra_peak]:
        between = smoothed[intra_peak:later_peak + 1]
        if between.size == 0:
            continue
        valley_offset = int(np.argmin(between))
        valley_height = between[valley_offset]
        peak_heights = smoothed[intra_peak] * smoothed[later_peak]
        if peak_heights <= 0:
            continue
        # Void parameter (Pasquale et al. 2010): how deep the valley is
        # relative to the two peaks it separates. 0 means no separation.
        void = 1.0 - valley_height / np.sqrt(peak_heights)
        if void >= options["min_void"]:
            threshold = float(10 ** centers[intra_peak + valley_offset])
            return min(threshold, options["cutoff_s"]), "antimode"

    return options["cutoff_s"], "cutoff"


def logisi_bursts(times, threshold_s=None, **params):
    """logISI burst detection. Returns inclusive (start, end) index pairs.

    Note: this implements the threshold selection and the grouping of spikes
    below it. Pasquale's optional second pass, which re-attaches
    "burst-related" spikes out to the longer inter-burst threshold, is not
    implemented; it lengthens bursts slightly and does not change burst counts.
    """
    options = {**LOGISI_DEFAULTS, **params}
    times = np.sort(np.asarray(times, dtype=float))
    if times.size < options["min_spikes"]:
        return []

    if threshold_s is None:
        threshold_s, _ = logisi_threshold(times, **params)

    isis = np.diff(times)
    bursts = []
    start = None
    for index, isi in enumerate(isis):
        if isi <= threshold_s:
            if start is None:
                start = index
        elif start is not None:
            bursts.append((start, index))
            start = None
    if start is not None:
        bursts.append((start, len(isis)))

    return [
        (start, end) for start, end in bursts
        if (end - start + 1) >= options["min_spikes"]
    ]


# ---------------------------------------------------------------------------
# Per-unit metrics
# ---------------------------------------------------------------------------

def _cv(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    mean = values.mean()
    return float(values.std() / mean) if abs(mean) > 1e-12 else None


def burst_metrics(times, bursts, duration_s):
    """Metrics for one unit given its detected bursts.

    Every rate divides by the recording duration, so a unit that bursts only
    in the first minute of a ten minute recording is not credited with the
    burst rate of a unit that bursts throughout.
    """
    times = np.sort(np.asarray(times, dtype=float))
    n_spikes = int(times.size)

    metrics = {
        "n_spikes": n_spikes,
        "n_bursts": len(bursts),
        "burst_rate_hz": (len(bursts) / duration_s) if duration_s > 0 else None,
        "fraction_spikes_in_bursts": None,
        "burst_duration_mean_s": None,
        "burst_duration_cv": None,
        "spikes_per_burst_mean": None,
        "intraburst_rate_hz": None,
        "ibi_mean_s": None,
        "ibi_cv": None,
        "ibi_gap_mean_s": None,
        "is_bursting": False,
    }

    if not bursts:
        metrics["fraction_spikes_in_bursts"] = 0.0
        return metrics

    starts = np.array([times[start] for start, _ in bursts], dtype=float)
    ends = np.array([times[end] for _, end in bursts], dtype=float)
    durations = ends - starts
    spike_counts = np.array([end - start + 1 for start, end in bursts], dtype=float)

    metrics["fraction_spikes_in_bursts"] = float(spike_counts.sum() / n_spikes) if n_spikes else 0.0
    metrics["burst_duration_mean_s"] = float(durations.mean())
    metrics["burst_duration_cv"] = _cv(durations)
    metrics["spikes_per_burst_mean"] = float(spike_counts.mean())

    # Intra-burst rate per burst, then averaged, so long bursts do not dominate.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_burst_rate = np.where(durations > 0, spike_counts / durations, np.nan)
    if np.any(np.isfinite(per_burst_rate)):
        metrics["intraburst_rate_hz"] = float(np.nanmean(per_burst_rate))

    if len(starts) > 1:
        metrics["ibi_mean_s"] = float(np.mean(np.diff(starts)))
        metrics["ibi_cv"] = _cv(np.diff(starts))
        metrics["ibi_gap_mean_s"] = float(np.mean(np.clip(starts[1:] - ends[:-1], 0.0, None)))

    # "Bursting" needs enough bursts for the interval statistics to mean
    # anything; two bursts give one interval and no variability estimate.
    metrics["is_bursting"] = bool(len(bursts) >= 3)
    return metrics


def _in_burst_mask(n_spikes, bursts):
    mask = np.zeros(n_spikes, dtype=bool)
    for start, end in bursts:
        mask[start:end + 1] = True
    return mask


def unit_burst_features(times, duration_s, maxinterval_params=None, logisi_params=None):
    """Both detectors plus their agreement, for one unit. Flat scalar dict."""
    times = np.sort(np.asarray(times, dtype=float))
    maxinterval_params = maxinterval_params or {}
    logisi_params = logisi_params or {}

    mi_bursts = maxinterval_bursts(times, **maxinterval_params)
    threshold_s, threshold_source = logisi_threshold(times, **logisi_params)
    li_bursts = logisi_bursts(times, threshold_s=threshold_s, **logisi_params)

    features = {
        "n_spikes": int(times.size),
        "mean_firing_rate_hz": (float(times.size / duration_s) if duration_s > 0 else None),
        "logisi_threshold_s": float(threshold_s),
        "logisi_threshold_source": threshold_source,
    }
    for prefix, bursts in (("mi", mi_bursts), ("li", li_bursts)):
        for key, value in burst_metrics(times, bursts, duration_s).items():
            if key == "n_spikes":
                continue
            features[f"{prefix}_{key}"] = value

    # How far the two detectors agree on which spikes are in bursts. Cotterill
    # et al. 2016 note that where methods disagree, human scorers tend to
    # disagree too, so a low value flags units whose bursting is ambiguous
    # rather than a bug in either detector.
    if times.size:
        mi_mask = _in_burst_mask(times.size, mi_bursts)
        li_mask = _in_burst_mask(times.size, li_bursts)
        union = int(np.count_nonzero(mi_mask | li_mask))
        features["agreement_jaccard"] = (
            float(np.count_nonzero(mi_mask & li_mask) / union) if union else None
        )
    else:
        features["agreement_jaccard"] = None

    return features


# ---------------------------------------------------------------------------
# Population summary
# ---------------------------------------------------------------------------

# Per-unit fields worth summarising across units for the well-level table.
SUMMARY_FIELDS = (
    "mean_firing_rate_hz",
    "mi_burst_rate_hz", "mi_burst_duration_mean_s", "mi_spikes_per_burst_mean",
    "mi_intraburst_rate_hz", "mi_fraction_spikes_in_bursts", "mi_ibi_mean_s", "mi_ibi_cv",
    "li_burst_rate_hz", "li_burst_duration_mean_s", "li_spikes_per_burst_mean",
    "li_intraburst_rate_hz", "li_fraction_spikes_in_bursts", "li_ibi_mean_s", "li_ibi_cv",
    "agreement_jaccard",
)


def _summarise(values):
    """mean/median/std/cv over the units that had a defined value."""
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "cv": None}
    mean = float(finite.mean())
    std = float(finite.std())
    return {
        "n": int(finite.size),
        "mean": mean,
        "median": float(np.median(finite)),
        "std": std,
        "cv": (std / mean) if abs(mean) > 1e-12 else None,
    }


def compute_unit_burst_features(SpikeTimes, duration_s=None,
                                maxinterval_params=None, logisi_params=None):
    """Per-unit burst features for a whole well.

    Returns {"units": {unit_id: flat dict}, "summary": {...}, "params": {...}}.
    `duration_s` should be the recording duration; without it the span between
    the first and last spike of each unit is used, which inflates the rates of
    units that are active for only part of the recording.
    """
    units = list(SpikeTimes.keys())
    if not units:
        return {"units": {}, "summary": {"n_units": 0}, "params": {}}

    if duration_s is None or float(duration_s) <= 0:
        all_times = [np.asarray(SpikeTimes[u], dtype=float) for u in units]
        non_empty = [t for t in all_times if t.size]
        if not non_empty:
            return {"units": {}, "summary": {"n_units": 0}, "params": {}}
        pooled = np.concatenate(non_empty)
        duration_s = float(pooled.max() - pooled.min())
        duration_source = "spike_span"
    else:
        duration_s = float(duration_s)
        duration_source = "recording"

    if duration_s <= 0:
        return {"units": {}, "summary": {"n_units": 0}, "params": {}}

    per_unit = {}
    for unit_id in units:
        times = np.asarray(SpikeTimes[unit_id], dtype=float)
        if times.size == 0:
            continue
        per_unit[unit_id] = unit_burst_features(
            times, duration_s,
            maxinterval_params=maxinterval_params,
            logisi_params=logisi_params,
        )

    summary = {
        "n_units": len(per_unit),
        "duration_s": duration_s,
        "duration_source": duration_source,
        "n_bursting_units_maxinterval": sum(
            1 for f in per_unit.values() if f.get("mi_is_bursting")
        ),
        "n_bursting_units_logisi": sum(
            1 for f in per_unit.values() if f.get("li_is_bursting")
        ),
        "n_units_logisi_adaptive_threshold": sum(
            1 for f in per_unit.values() if f.get("logisi_threshold_source") == "antimode"
        ),
    }
    for field in SUMMARY_FIELDS:
        summary[field] = _summarise([f.get(field) for f in per_unit.values()])

    return {
        "units": per_unit,
        "summary": summary,
        "params": {
            "maxinterval": {**MAXINTERVAL_DEFAULTS, **(maxinterval_params or {})},
            "logisi": {**LOGISI_DEFAULTS, **(logisi_params or {})},
        },
    }
