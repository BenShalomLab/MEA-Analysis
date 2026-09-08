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
#   - Peak detection mirrors MATLAB's default GUI threshold method,
#     ThresholdMethod='Adaptive', AS ACTUALLY SHIPPED in
#     computeNetworkStatsModified.m — that branch's intended mean+2*SD
#     rolling-threshold logic is commented out/dead code there, so no
#     height/amplitude gate is applied at all. Peaks are constrained by
#     MinPeakProminence and MinPeakDistance only (findpeaks(...,
#     peakParams{:}) with no 'MinPeakHeight'). NOT the same as the 'Fixed'
#     (MinPeakHeight=Threshold) or 'RMS' (MinPeakHeight=Threshold*rms(...))
#     branches of that same MATLAB function, which do apply a height gate.
#   - Burst onset/offset = walk outward from each peak until the rate drops
#     below a fixed fraction of the peak value (percentage-of-peak edges,
#     MATLAB's thresholdStartStop in gaussianFiringRateBurstDetector).
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
    from burst_common import level_metrics
except ImportError:
    from MEA_Analysis.IPNAnalysis.burst_common import level_metrics

try:
    from catch22_features import compute_catch22, compute_unit_catch22_features
except ImportError:
    from MEA_Analysis.IPNAnalysis.catch22_features import compute_catch22, compute_unit_catch22_features


def compute_network_bursts(
    SpikeTimes=None,
    bin_size_s=0.01,
    gaussian_sigma_s=0.1,
    min_prominence=None,
    min_peak_distance_s=1.0,
    onset_offset_peak_frac=0.3,
    compute_catch22_features=False,
):
    """Gaussian population-rate network burst detector.

    Parameters
    ----------
    SpikeTimes : dict[unit_id, array-like]
        Spike times in seconds, per unit.
    bin_size_s : float
        Histogram bin width (s). Mirrors MATLAB BinSize.
    gaussian_sigma_s : float
        Gaussian smoothing kernel sigma (s). Mirrors MATLAB GaussianSigma.
    min_prominence : float or None
        Minimum peak prominence (Hz/unit), required for a candidate peak
        to count — mirrors MATLAB's 'MinPeakProminence'. None (default)
        derives it from the signal itself as baseline_std (informational
        only — see note on ThresholdMethod='Adaptive' above, there is no
        height/amplitude threshold applied here, by design, to match
        current MATLAB behavior); pass an explicit Hz/unit value to mirror
        a specific MATLAB MinPeakProminence setting.
    min_peak_distance_s : float
        Minimum spacing between detected burst peaks (s). Mirrors MATLAB
        MinPeakDistance.
    onset_offset_peak_frac : float
        Burst edges are where the smoothed rate first drops below
        peak_value * (1 - onset_offset_peak_frac), walking outward from
        the peak. Mirrors MATLAB thresholdStartStop in
        gaussianFiringRateBurstDetector.
    compute_catch22_features : bool
        Off by default (no change to schema/runtime cost unless opted in).
        When True, adds a "catch22_features" block to the return value:
        {"population": {...}, "unit_features": {unit_id: {...}}} — 22
        canonical time-series characteristics (github.com/
        DynamicsAndNeuralSystems/catch22) computed on the smoothed
        population-rate trace and on each unit's binned spike train,
        respectively. See catch22_features.py.

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
    total_dur = rec_end - rec_start
    if total_dur <= 0:
        return {"error": "no_spikes"}

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
    participation_fraction_signal = active_unit_counts / max(1, n_units)

    # ---------------------------------------------------------
    # 3. Peak detection (on smoothed rate) — prominence + distance only,
    # no height/amplitude gate. See module docstring: this mirrors MATLAB's
    # ThresholdMethod='Adaptive' as actually shipped (its intended
    # mean+2*SD rolling threshold is dead/commented-out code there).
    # baseline_mean/std are kept in diagnostics for reference only — they
    # are NOT used to gate peak detection.
    # ---------------------------------------------------------
    baseline_mean = float(np.mean(smoothed_rate_hz))
    baseline_std = float(np.std(smoothed_rate_hz))
    effective_min_prominence = float(min_prominence) if min_prominence is not None else baseline_std
    min_distance_bins = max(1, int(min_peak_distance_s / bin_size_s))

    peaks, _ = find_peaks(
        smoothed_rate_hz,
        prominence=effective_min_prominence,
        distance=min_distance_bins,
    )

    network_bursts = []
    if len(peaks) > 0:
        for peak_idx in peaks:
            peak_val = smoothed_rate_hz[peak_idx]
            edge_level = peak_val * (1 - onset_offset_peak_frac)

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
            spike_count = int(np.sum(in_burst))
            participating_units = len(np.unique(all_spike_units[in_burst])) if spike_count else 0
            participation_fraction = participating_units / n_units if n_units else 0.0

            network_bursts.append({
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "burst_duration_s": burst_duration_s,
                "peak_time_s": float(t_centers[peak_idx]),
                "peak_population_firing_rate_hz": float(peak_val),
                "burst_area": float(np.sum(smoothed_rate_hz[i:j + 1]) * bin_size_s),
                "spike_count": spike_count,
                "participation_fraction": float(participation_fraction),
                "peak_participation_fraction": float(participation_fraction_signal[peak_idx]),
            })

    # ---------------------------------------------------------
    # 3b. catch22 features (opt-in, see compute_catch22_features docstring)
    # ---------------------------------------------------------
    catch22_block = None
    if compute_catch22_features:
        catch22_block = {
            "population": compute_catch22(smoothed_rate_hz),
            "unit_features": compute_unit_catch22_features(
                SpikeTimes, bin_size_s, rec_start, rec_end
            ),
        }

    # ---------------------------------------------------------
    # 4. Assemble return value (schema-compatible, single tier)
    # ---------------------------------------------------------
    result = {
        "burst_fragments": {"events": [], "metrics": {}},
        "network_bursts": {
            "events": network_bursts,
            "metrics": level_metrics(network_bursts, total_dur, ibi_key="ibi_s"),
        },
        "superbursts": {"events": [], "metrics": {}},

        "diagnostics": {
            "method": "gaussian_population_rate_adaptive_no_height_gate",
            "bin_size_ms": bin_size_s * 1000.0,
            "gaussian_sigma_s": gaussian_sigma_s,
            # Reference only — NOT used to gate peak detection. Mirrors
            # MATLAB ThresholdMethod='Adaptive' as shipped: no height
            # threshold, prominence + distance only (see module docstring).
            "baseline_mean_hz": baseline_mean,
            "baseline_std_hz": baseline_std,
            "min_prominence_hz": effective_min_prominence,
            "min_peak_distance_s": min_peak_distance_s,
            "onset_offset_peak_frac": onset_offset_peak_frac,
            "n_units": n_units,
            "burst_detection_valid": True,
        },

        "unit_stats": unit_stats,

        "plot_data": {
            "time_s": t_centers,
            "participation_fraction_signal": participation_fraction_signal,
            "population_firing_rate_hz": smoothed_rate_hz,
            "nb_peak_times_s": np.array([b["peak_time_s"] for b in network_bursts]),
            "nb_peak_participation_fraction": np.array([b["peak_participation_fraction"] for b in network_bursts]),
            "sb_start_times_s": np.array([]),
            "sb_end_times_s": np.array([]),
            "participation_baseline": float(np.mean(participation_fraction_signal)),
            "detection_threshold": None,  # threshold is in Hz, not participation-fraction space; see diagnostics.detection_threshold_hz
        },
    }

    if catch22_block is not None:
        result["catch22_features"] = catch22_block

    return result
