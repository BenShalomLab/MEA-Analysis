# ==========================================================
# gaussianNetworkBursts.py
# Author: Mandar Patil
# LLM Assisted Edits: Yes Claude Sonnet 5
#
# Gaussian-kernel population-rate network burst detector.
#
# Ports the lab's MATLAB pipeline (mxw.networkActivity.computeNetworkAct +
# computeNetworkStatsModified + gaussianFiringRateBurstDetector), not the
# adaptive/parameter-free method in parameter_free_burst_detector.py:
#   - Pool all spikes into a population firing-rate histogram, Gaussian-
#     smooth it (Wagenaar et al. 2006; Chiappalone et al. 2005), normalized
#     per active unit/electrode to match mxw.networkActivity.computeNetworkAct.
#   - Peak detection applies a mean + N*SD height gate (Chiappalone et al.
#     2005; also the intended behaviour of ThresholdMethod='Adaptive' in
#     computeNetworkStatsModified.m, where that branch's mean+2*SD rolling
#     threshold is commented out and therefore never runs). Reproducing the
#     MATLAB code as shipped — prominence and distance only, no height gate
#     — means a silent well still yields "network bursts", because the
#     prominence floor derived from that well's own SD shrinks with it.
#     Set min_height_sd=None to restore the ungated MATLAB behaviour.
#   - Burst onset/offset = walk outward from each peak until the rate drops
#     below onset_offset_peak_frac * peak (percentage-of-peak edges, MATLAB's
#     thresholdStartStop in gaussianFiringRateBurstDetector; Wagenaar et al.
#     2006 use the same construction).
#
# Parameter names mirror the MATLAB call sites 1:1 so config values can be
# copied across directly:
#   gaussian_sigma_s   <-> GaussianSigma      (computeNetworkAct)
#   bin_size_s         <-> BinSize            (computeNetworkAct)
#   min_prominence      <-> MinPeakProminence (computeNetworkStatsModified)
#   min_peak_distance_s <-> MinPeakDistance   (computeNetworkStatsModified)
#   onset_offset_peak_frac <-> thresholdStartStop (gaussianFiringRateBurstDetector)
#
# Single-tier detector: only "network_bursts" are produced directly from
# peaks in the smoothed rate signal — there is no fragment-extraction or
# superburst-merge stage (contrast with parameter_free_burst_detector.py's
# three-tier fragment -> network burst -> superburst hierarchy).
# "burst_fragments" and "superbursts" are still present as empty tiers so
# the return value is schema-compatible with compute_network_bursts() in
# parameter_free_burst_detector.py and can be swapped in as a drop-in
# alternative in mea_reports.py.
# ==========================================================

import numpy as np
from scipy.signal import find_peaks, convolve

try:
    from burst_common import (level_metrics, SCHEMA_VERSION, spike_participation,
                              intraburst_isi_mean as _intraburst_isi_mean)
except ImportError:
    from MEA_Analysis.IPNAnalysis.burst_common import (
        level_metrics, SCHEMA_VERSION, spike_participation,
        intraburst_isi_mean as _intraburst_isi_mean)


def compute_network_bursts(
    SpikeTimes=None,
    duration_s=None,
    bin_size_s=0.01,
    gaussian_sigma_s=0.1,
    min_prominence=None,
    min_peak_distance_s=1.0,
    onset_offset_peak_frac=0.3,
    min_height_sd=2.0,
):
    """Gaussian population-rate network burst detector.

    Parameters
    ----------
    SpikeTimes : dict[unit_id, array-like]
        Spike times in seconds, per unit.
    duration_s : float or None
        Duration of the *recording* in seconds, used as the denominator for
        all firing and burst rates. None falls back to the span between the
        first and last spike, which inflates every rate in a well that is
        silent for part of the recording.
    bin_size_s : float
        Histogram bin width (s). Mirrors MATLAB BinSize.
    gaussian_sigma_s : float
        Gaussian smoothing kernel sigma (s). Mirrors MATLAB GaussianSigma.
    min_prominence : float or None
        Minimum peak prominence (Hz/unit), required for a candidate peak
        to count — mirrors MATLAB's 'MinPeakProminence'. None (default)
        derives it from the signal itself as its standard deviation; pass an
        explicit Hz/unit value to mirror a specific MATLAB
        MinPeakProminence setting.
    min_peak_distance_s : float
        Minimum spacing between detected burst peaks (s). Mirrors MATLAB
        MinPeakDistance.
    onset_offset_peak_frac : float
        Burst edges are where the smoothed rate first drops below
        peak_value * onset_offset_peak_frac, walking outward from the peak.
        Mirrors MATLAB thresholdStartStop in
        gaussianFiringRateBurstDetector: 0.3 means "edges at 30% of peak
        height", so a smaller value gives longer bursts.
    min_height_sd : float or None
        Height gate for peak detection, in standard deviations of the
        smoothed rate above its mean (Chiappalone et al. 2005 use mean + N*SD
        with N of 2 or higher). None disables the gate, reproducing the
        MATLAB code's shipped ThresholdMethod='Adaptive' behaviour, in which
        a silent well still produces "network bursts" from noise ripples.

    Returns
    -------
    dict with the same top-level schema as
    parameter_free_burst_detector.compute_network_bursts: "burst_fragments",
    "network_bursts", "superbursts" (each {"events": [...], "metrics": {...}}),
    "diagnostics", "unit_stats", "plot_data". On failure, returns
    {"error": "no_units" | "no_spikes"}.
    """
    if SpikeTimes is None:
        SpikeTimes = {}

    units = list(SpikeTimes.keys())
    if not units:
        return {"error": "no_units"}

    # ---------------------------------------------------------
    # 1. Flatten spikes, track per-unit firing rate
    # ---------------------------------------------------------
    all_spike_times = []
    all_spike_units = []
    unit_stats = {}

    non_empty = [SpikeTimes[u] for u in units if len(SpikeTimes[u]) > 0]
    if not non_empty:
        return {"error": "no_spikes"}

    all_spikes_sorted = np.sort(np.concatenate([np.asarray(s) for s in non_empty]))
    rec_start = float(all_spikes_sorted[0])
    rec_end = float(all_spikes_sorted[-1])

    # Binning spans the active period; rates are divided by the recording
    # duration when it is known (see duration_s in the docstring).
    analysis_window_s = rec_end - rec_start
    if analysis_window_s <= 0:
        return {"error": "no_spikes"}
    if duration_s is not None and float(duration_s) > 0:
        total_dur = float(duration_s)
        duration_source = "recording"
    else:
        total_dur = analysis_window_s
        duration_source = "spike_span"

    for u in units:
        t = np.asarray(SpikeTimes[u])
        if t.size == 0:
            unit_stats[u] = {"mean_firing_rate_hz": 0.0}
            continue
        all_spike_times.append(t)
        all_spike_units.extend([u] * len(t))
        unit_stats[u] = {"mean_firing_rate_hz": float(len(t) / total_dur)}

    all_spike_times = np.concatenate(all_spike_times)
    all_spike_units = np.array(all_spike_units)

    sort_idx = np.argsort(all_spike_times)
    all_spike_times = all_spike_times[sort_idx]
    all_spike_units = all_spike_units[sort_idx]

    n_units = sum(1 for u in units if len(SpikeTimes[u]) > 0)

    # ---------------------------------------------------------
    # 2. Population rate histogram + Gaussian smoothing
    # ---------------------------------------------------------
    bins = np.arange(rec_start, rec_end + bin_size_s, bin_size_s)
    t_centers = bins[:-1] + bin_size_s / 2

    spike_counts, _ = np.histogram(all_spike_times, bins=bins)
    # Mean firing rate per active unit/electrode (Hz/unit) — matches MATLAB's
    # mxw.networkActivity.computeNetworkAct convention and the sibling
    # detector's rate_signal_raw (parameter_free_burst_detector.py), NOT a
    # raw pooled sum across all units. Omitting the /n_units division here
    # previously inflated the signal (and every threshold/prominence value
    # derived from it) by a factor of n_units.
    population_firing_rate_hz = spike_counts / bin_size_s / max(1, n_units)

    kernel_half_width = np.arange(-4 * gaussian_sigma_s, 4 * gaussian_sigma_s + bin_size_s, bin_size_s)
    kernel = np.exp(-0.5 * (kernel_half_width / gaussian_sigma_s) ** 2)
    kernel /= kernel.sum()

    smoothed_rate_hz = convolve(population_firing_rate_hz, kernel, mode="same")

    # Participation fraction is not the detection signal here (rate is, per
    # the literature method above) but is computed for downstream plotting
    # via helper.plot_clean_network, which draws it as the primary trace.
    active_unit_counts = np.zeros(len(t_centers))
    for u in units:
        spk = np.asarray(SpikeTimes[u])
        if spk.size == 0:
            continue
        counts, _ = np.histogram(spk, bins=bins)
        active_unit_counts += counts > 0
    coactive_fraction_signal = active_unit_counts / max(1, n_units)

    # ---------------------------------------------------------
    # 3. Peak detection (on smoothed rate): height gate + prominence +
    # distance. The height gate is mean + min_height_sd * SD of the smoothed
    # rate (Chiappalone et al. 2005). Note both statistics are taken over the
    # whole trace including the bursts, so they rise with burst load — this
    # is the literature convention and is deliberately kept, but it means the
    # threshold is not a pure baseline estimate.
    # ---------------------------------------------------------
    baseline_mean = float(np.mean(smoothed_rate_hz))
    baseline_std = float(np.std(smoothed_rate_hz))
    effective_min_prominence = float(min_prominence) if min_prominence is not None else baseline_std
    min_distance_bins = max(1, int(min_peak_distance_s / bin_size_s))

    # None or a non-positive value disables the gate (MATLAB parity).
    if min_height_sd is None or float(min_height_sd) <= 0:
        min_height_sd = None
        detection_threshold_hz = None
    else:
        detection_threshold_hz = baseline_mean + float(min_height_sd) * baseline_std

    peaks, _ = find_peaks(
        smoothed_rate_hz,
        height=detection_threshold_hz,
        prominence=effective_min_prominence,
        distance=min_distance_bins,
    )

    network_bursts = []
    if len(peaks) > 0:
        for peak_idx in peaks:
            peak_val = smoothed_rate_hz[peak_idx]
            # Percentage-of-peak edges: walk out until the rate falls below
            # this fraction OF the peak. (An earlier version used
            # peak * (1 - frac), i.e. edges at 70% of peak for frac=0.3,
            # which clipped every burst to its crest and under-reported
            # burst duration several-fold.)
            edge_level = peak_val * onset_offset_peak_frac

            i = peak_idx
            while i > 0 and smoothed_rate_hz[i] > edge_level:
                i -= 1
            j = peak_idx
            while j < len(smoothed_rate_hz) - 1 and smoothed_rate_hz[j] > edge_level:
                j += 1

            start_time_s = float(t_centers[i])
            end_time_s = float(t_centers[j])
            burst_duration_s = end_time_s - start_time_s
            if burst_duration_s <= 0:
                continue

            in_burst = (all_spike_times >= start_time_s) & (all_spike_times <= end_time_s)
            spikes_per_burst = int(np.sum(in_burst))
            participating_units = len(np.unique(all_spike_units[in_burst])) if spikes_per_burst else 0
            participation_fraction = participating_units / n_units if n_units else 0.0

            denom = burst_duration_s * max(1, participating_units)
            peak_time_s = float(t_centers[peak_idx])
            network_bursts.append({
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "burst_duration_s": burst_duration_s,
                "peak_time_s": peak_time_s,
                # RT and DT of Mossink et al. 2021.
                "rise_time_s": max(0.0, peak_time_s - start_time_s),
                "decay_time_s": max(0.0, end_time_s - peak_time_s),
                # Per unit, as in the sibling detector: the signal this peak is
                # read from is already divided by n_units.
                "burst_peak_hz_per_unit": float(peak_val),
                # Array-wide equivalent, recovered by undoing that division.
                # Kept so the key exists at the same meaning in both detectors.
                "burst_peak_hz_array": float(peak_val * max(1, n_units)),
                "burst_area_spikes_per_unit": float(np.sum(smoothed_rate_hz[i:j + 1]) * bin_size_s),
                "spikes_per_burst": spikes_per_burst,
                "spikes_per_burst_per_unit": float(spikes_per_burst / max(1, n_units)),
                "intraburst_rate_hz": float(spikes_per_burst / denom) if denom > 0 else 0.0,
                "participation_fraction": float(participation_fraction),
                "coactive_fraction_peak": float(coactive_fraction_signal[peak_idx]),
                "coactive_fraction_max": float(np.max(coactive_fraction_signal[i:j + 1])),
                "intraburst_isi_mean_s": _intraburst_isi_mean(
                    all_spike_times, start_time_s, end_time_s),
            })

    # ---------------------------------------------------------
    # 4. Assemble return value (schema-compatible, single tier)
    # ---------------------------------------------------------
    return {
        "burst_fragments": {"events": [], "metrics": {}},
        "network_bursts": {
            "events": network_bursts,
            "metrics": level_metrics(network_bursts, total_dur, ibi_key="ibi_s"),
        },
        "superbursts": {"events": [], "metrics": {}},

        "spike_participation": spike_participation(all_spike_times, network_bursts),

        "diagnostics": {
            "schema_version": SCHEMA_VERSION,
            "detector": "gaussian",
            "method": (
                "gaussian_population_rate_mean_plus_sd"
                if min_height_sd is not None
                else "gaussian_population_rate_no_height_gate"
            ),
            "recording_duration_s": total_dur,
            "analysis_window_s": analysis_window_s,
            "duration_source": duration_source,
            "bin_size_ms": bin_size_s * 1000.0,
            "gaussian_sigma_s": gaussian_sigma_s,
            "baseline_mean_hz": baseline_mean,
            "baseline_std_hz": baseline_std,
            "min_height_sd": min_height_sd,
            "detection_threshold_hz": detection_threshold_hz,
            "min_prominence_hz": effective_min_prominence,
            "min_peak_distance_s": min_peak_distance_s,
            "onset_offset_peak_frac": onset_offset_peak_frac,
            "edge_rule": "fraction_of_peak",
            "n_units": n_units,
            "burst_detection_valid": True,
        },

        "unit_stats": unit_stats,

        "plot_data": {
            "time_s": t_centers,
            "coactive_fraction_signal": coactive_fraction_signal,
            "population_firing_rate_hz": smoothed_rate_hz,
            "nb_peak_times_s": np.array([b["peak_time_s"] for b in network_bursts]),
            "nb_peak_coactive_fraction": np.array([b["coactive_fraction_peak"] for b in network_bursts]),
            "sb_start_times_s": np.array([]),
            "sb_end_times_s": np.array([]),
            "coactive_fraction_baseline": float(np.mean(coactive_fraction_signal)),
            "detection_threshold": None,  # threshold is in Hz, not participation-fraction space; see diagnostics.detection_threshold_hz
        },
    }
