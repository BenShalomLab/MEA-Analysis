import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis as sp_kurtosis

try:
    from burst_common import (stats, level_metrics as _level_metrics, SCHEMA_VERSION,
                              spike_participation,
                              intraburst_isi_mean as _intraburst_isi_mean)
except ImportError:
    from MEA_Analysis.IPNAnalysis.burst_common import (
        stats, level_metrics as _level_metrics, SCHEMA_VERSION,
        spike_participation, intraburst_isi_mean as _intraburst_isi_mean)


# Minimum number of network burst intervals needed before testing their
# distribution for the two modes that define superburst structure. Below this
# the histogram is noise and the test would find spurious antimodes.
MIN_IBIS_FOR_SUPERBURST_TEST = 8

# Gap used by merge_rule="gap", the comparison condition that merges on
# proximity alone. 2 s matches Parodi et al. 2026. Only reachable when the
# caller asks for that rule; the default rule uses no gap at all.
GAP_ONLY_DEFAULT_S = 2.0

# Sarle's bimodality coefficient above this is taken as two modes (SAS
# convention; also used for the per-unit log-ISI test above).
BIMODALITY_THRESHOLD = 0.555


def _otsu_split(values):
    """Threshold maximising between-class variance of a 1-D sample (Otsu 1979).

    Used to split the log network-burst intervals into "within a cluster" and
    "between clusters". Preferred over finding a valley in a histogram, which
    needs a bin count and a prominence in raw counts and so fails on skewed
    samples: fourteen intervals at 0.5 s against two at 20 s is unmistakably
    two groups, but leaves no histogram valley prominent enough to detect.

    Returns None when there is no spread to split.
    """
    x = np.sort(np.asarray(values, dtype=float))
    if x.size < 2 or x[0] == x[-1]:
        return None
    cumulative = np.cumsum(x)
    total = cumulative[-1]
    n = x.size
    k = np.arange(1, n)
    weight_low = k / n
    mean_low = cumulative[:-1] / k
    mean_high = (total - cumulative[:-1]) / (n - k)
    between = weight_low * (1.0 - weight_low) * (mean_low - mean_high) ** 2
    best = int(np.argmax(between))
    return float((x[best] + x[best + 1]) / 2.0)


def compute_network_bursts(
    SpikeTimes=None,
    duration_s=None,
    extent_frac=0.30,
    network_merge_gap_min=None,
    fragment_max_gap_s=None,
    merge_rule="continuity",
    threshold_mad_scale=0.75,
    min_fragment_participation=0.0,
    min_burst_density_Hz=0.0,
    min_absolute_rate_Hz=0.0,
    min_superburst_dur_s=2.5,
    min_superburst_components=2,
    min_peak_synchrony=0.05,
):
    """Adaptive, three-tier network burst detector.

    Parameters
    ----------
    SpikeTimes : dict[unit_id, array-like]
        Spike times in seconds, per unit.
    duration_s : float or None
        Duration of the *recording* in seconds. All firing and burst rates
        are divided by this. When None it falls back to the span between the
        first and last spike, which overestimates every rate in a well that
        is silent for part of the recording — pass the real duration
        (`recording.get_num_frames() / fs`) whenever it is known.
        Detection itself still operates on the spike span, so passing this
        changes reported rates but not which bursts are found.
    min_superburst_components : int
        Minimum number of component network bursts for a superburst.
        Default 2: a superburst is a *cluster* of network bursts (Wagenaar
        et al. 2006). A single long network burst is reported instead
        through `network_bursts.metrics.burst_duration_p95_s`.
    """

    # ---------------------------------------------------------
    # 0. Sanity checks
    # ---------------------------------------------------------
    units = list(SpikeTimes.keys())
    if not units:
        return {"error": "no_units"}

    non_empty = [SpikeTimes[u] for u in units if len(SpikeTimes[u]) > 0]
    if not non_empty:
        return {"error": "no_spikes"}
    all_spikes = np.sort(np.concatenate(non_empty))
    if all_spikes.size == 0:
        return {"error": "no_spikes"}

    rec_start = float(all_spikes[0])
    rec_end   = float(all_spikes[-1])

    # Binning/detection window: the active span. Rate denominator: the whole
    # recording. Keeping these separate means a partially silent well is not
    # credited with a higher rate than a continuously active one, while
    # detection thresholds are still estimated from the period that has data.
    analysis_window_s = rec_end - rec_start
    if duration_s is not None and float(duration_s) > 0:
        total_dur = float(duration_s)
        duration_source = "recording"
    else:
        total_dur = analysis_window_s
        duration_source = "spike_span"
    if total_dur <= 0:
        return {"error": "no_spikes"}

    # ---------------------------------------------------------
    # 1. Biological calibration
    # ---------------------------------------------------------
    all_log_isis    = []
    bursty_log_isis = []
    unit_stats      = {}

    for u in units:
        t = np.unique(np.sort(SpikeTimes[u]))
        if len(t) < 2:
            unit_stats[u] = {"mean_firing_rate_hz": len(SpikeTimes[u]) / total_dur}
            continue

        isi = np.diff(t)
        isi = isi[isi > 0]
        if isi.size == 0:
            unit_stats[u] = {"mean_firing_rate_hz": len(t) / total_dur}
            continue

        log_isi = np.log10(isi)
        all_log_isis.extend(log_isi)

        mean_fr = len(t) / total_dur
        isi_cv  = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else np.nan

        # CV2 — local irregularity, robust to rate non-stationarity (Holt 1996)
        if len(isi) >= 2:
            cv2 = float(np.mean(2 * np.abs(np.diff(isi)) / (isi[:-1] + isi[1:])))
        else:
            cv2 = np.nan

        # Lv — local variation (Shinomoto 2009)
        if len(isi) >= 2:
            lv = float(3 * np.mean(((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:]))**2))
        else:
            lv = np.nan

        # Bimodality coefficient on log-ISI (Sarle 1990)
        n = len(log_isi)
        if n >= 4:
            g1 = skew(log_isi)
            g2 = sp_kurtosis(log_isi, fisher=True)
            bc = (g1**2 + 1) / (g2 + 3 * ((n - 1)**2 / ((n - 2) * (n - 3))))
        else:
            bc = np.nan

        is_bursty_by_isi_statistics = bool((not np.isnan(bc)) and bc > 0.555 and (np.isnan(lv) or lv > 1.0))

        unit_stats[u] = {
            "mean_firing_rate_hz":    mean_fr,
            "isi_cv":                 isi_cv,
            "cv2":                    cv2,
            "lv":                     lv,
            "log_isi_bimodality": float(bc) if not np.isnan(bc) else None,
            "is_bursty_by_isi_statistics":              is_bursty_by_isi_statistics,
        }

        if is_bursty_by_isi_statistics:
            bursty_log_isis.extend(log_isi)

    if len(bursty_log_isis) > 50:
        hist, edges = np.histogram(bursty_log_isis, bins=100)
        centers     = (edges[:-1] + edges[1:]) / 2
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        peaks, _    = find_peaks(hist_smooth, prominence=5)
        if len(peaks) > 0:
            reference_isi_s = float(10 ** centers[peaks[0]])
        else:
            reference_isi_s = float(10 ** np.percentile(bursty_log_isis, 15))
    elif all_log_isis:
        reference_isi_s = float(10 ** np.percentile(all_log_isis, 15))
    else:
        reference_isi_s = 0.05

    # Bin size: 20ms floor, 100ms ceiling (Chiappalone et al. 2005)
    bin_size_ms = np.clip(reference_isi_s * 1000, 20, 100)
    bin_size    = bin_size_ms / 1000.0

    bins      = np.arange(rec_start, rec_end + bin_size, bin_size)
    t_centers = (bins[:-1] + bins[1:]) / 2

    # ---------------------------------------------------------
    # 2. Population signals
    # ---------------------------------------------------------
    n_bins  = len(t_centers)
    n_units = sum(1 for u in units if len(SpikeTimes[u]) > 0)

    active_unit_counts = np.zeros(n_bins)
    spike_counts_total = np.zeros(n_bins)

    for u in units:
        spk = np.asarray(SpikeTimes[u])
        if spk.size == 0:
            continue

        counts, _ = np.histogram(spk, bins=bins)
        active_unit_counts += (counts > 0)
        spike_counts_total += counts

    coactive_fraction_raw = active_unit_counts / max(1, n_units)
    rate_signal_raw                   = spike_counts_total / bin_size / max(1, n_units)

    population_firing_rate_hz = spike_counts_total / bin_size

    # ---------------------------------------------------------
    # 3. Smoothing
    # ---------------------------------------------------------
    isi_bins = reference_isi_s / bin_size

    sigma_coactivity_bins = np.clip(isi_bins, 1, 2)
    sigma_firing_rate_bins   = np.clip(5.0 * isi_bins, 3, 8)

    coactive_fraction_signal = gaussian_filter1d(coactive_fraction_raw, sigma_coactivity_bins)
    population_firing_rate_signal = gaussian_filter1d(rate_signal_raw, sigma_firing_rate_bins)

    # ---------------------------------------------------------
    # 4. Detection thresholds
    # ---------------------------------------------------------
    coactive_fraction_baseline = np.median(coactive_fraction_signal)
    coactive_fraction_mad      = np.median(np.abs(coactive_fraction_signal - coactive_fraction_baseline))

    # Bimodality coefficient on participation signal (Sarle 1990).
    # BC selects threshold method — NOT used as a hard gate.
    # Sparse-burst recordings produce low BC even with genuine burst
    # structure because the burst population is too small relative to
    # the baseline to create a visible second mode. Hard gating on BC
    # causes false negatives in exactly these cases.
    _pf = coactive_fraction_signal
    _n  = len(_pf)
    if _n >= 4:
        _g1 = skew(_pf)
        _g2 = sp_kurtosis(_pf, fisher=True)
        coactive_fraction_bimodality = float(
            (_g1**2 + 1) / (_g2 + 3 * ((_n - 1)**2 / ((_n - 2) * (_n - 3))))
        )
    else:
        coactive_fraction_bimodality = 0.0

    if coactive_fraction_bimodality > 0.555:
        # Signal is genuinely bimodal — MAD robustly captures the
        # burst/baseline separation.
        detection_threshold = max(
            0.03,
            coactive_fraction_baseline + threshold_mad_scale * coactive_fraction_mad
        )
        threshold_source = "baseline_mad"
    else:
        # Sparse bursting in elevated or noisy baseline: MAD is small
        # because baseline dominates the distribution. Use a high percentile
        # of the participation signal to find the burst tail.
        #
        # Percentile scales with n_units: fewer units = noisier signal
        # (single-unit events create 1/n_units jumps) = need stricter
        # percentile to avoid trivially clearing threshold with noise.
        #   n_units=15  → pct ≈ 98.7  (very strict)
        #   n_units=100 → pct ≈ 97.3
        #   n_units=500 → pct ≈ 95.0  (standard)
        pct = float(np.clip(95.0 + 5.0 * np.exp(-n_units / 50.0), 95.0, 99.5))
        detection_threshold = max(
            0.03,
            float(np.percentile(coactive_fraction_signal, pct))
        )
        threshold_source = f"p{pct:.1f}"

    # Prominence: peaks must rise sharply above local surroundings.
    # 2*MAD filters broad low-amplitude elevations with no synchrony structure.
    min_prominence = max(2.0 * coactive_fraction_mad, 0.02)

    # Adaptive peak synchrony floor:
    # min_peak_synchrony as passed is interpreted as a fraction, but for
    # small cultures a single unit firing creates a 1/n_units jump that
    # trivially clears any fixed fraction floor.
    # Require at least max(3, 10% of n_units) units co-active in a single
    # bin — scales the absolute unit count floor with culture size.
    min_units_for_burst      = max(3, int(0.10 * n_units))
    min_coactive_fraction = max(
        min_peak_synchrony,
        min_units_for_burst / max(1, n_units)
    )

    # ---------------------------------------------------------
    # 5. Peak detection
    # ---------------------------------------------------------
    peaks, _ = find_peaks(
        coactive_fraction_signal,
        height=detection_threshold,
        prominence=min_prominence,
    )

    burst_fragments = []

    # ---------------------------------------------------------
    # 6. Fragment extraction
    # ---------------------------------------------------------
    for p in peaks:

        peak_val         = coactive_fraction_signal[p]
        extent_threshold = max(detection_threshold, extent_frac * peak_val)

        # LEFT boundary
        s = p
        while s > 0 and coactive_fraction_signal[s - 1] >= extent_threshold:
            s -= 1

        # RIGHT boundary
        e = p
        while e < n_bins - 1 and coactive_fraction_signal[e + 1] >= extent_threshold:
            e += 1

        start_idx = s
        end_idx   = e

        start_time_s = bins[start_idx]
        end_time_s   = bins[end_idx + 1]

        burst_duration_s = end_time_s - start_time_s
        if burst_duration_s <= 0:
            continue

        # Peak synchrony validation: require min_coactive_fraction
        # fraction of units co-active in a single bin within the fragment.
        # Adaptive floor prevents single-unit noise events from passing in
        # small cultures (n_units < 50) where 1/n_units jumps are large.
        coactive_fraction_max = float(
            np.max(active_unit_counts[start_idx:end_idx + 1]) / max(1, n_units)
        )
        if coactive_fraction_max < min_coactive_fraction:
            continue

        participating = sum(
            1 for u in units
            if np.any((SpikeTimes[u] >= start_time_s) & (SpikeTimes[u] < end_time_s))
        )

        participation_fraction = participating / n_units

        if min_fragment_participation > 0 and participation_fraction < min_fragment_participation:
            continue

        spikes_per_burst = int(np.sum(spike_counts_total[start_idx:end_idx + 1]))

        denom         = burst_duration_s * max(1, participating)
        burst_density = spikes_per_burst / denom if denom > 0 else 0

        peak_drive_rate = np.max(rate_signal_raw[start_idx:end_idx + 1])

        if min_burst_density_Hz > 0 and burst_density < min_burst_density_Hz:
            continue

        if min_absolute_rate_Hz > 0 and peak_drive_rate < min_absolute_rate_Hz:
            continue

        peak_time_s = float(t_centers[p])
        burst_fragments.append({
            "start_time_s":               float(start_time_s),
            "end_time_s":                 float(end_time_s),
            "burst_duration_s":           float(burst_duration_s),
            "peak_time_s":                peak_time_s,
            # Recruitment and termination, reported separately: Mossink et al.
            # 2021 (RT, DT) find these dissociate in disease models where burst
            # duration and rate do not.
            "rise_time_s":                max(0.0, peak_time_s - float(start_time_s)),
            "decay_time_s":               max(0.0, float(end_time_s) - peak_time_s),
            # Co-activity within a single bin at the peak: simultaneity. NOT
            # the maximum of participation_fraction, which is breadth over the
            # whole burst. Smoothed; coactive_fraction_max is the raw value.
            "coactive_fraction_peak":     float(peak_val),
            "coactive_fraction_max":      coactive_fraction_max,
            # Breadth of recruitment: "network burst participation" in the
            # literature (Bakkum et al. 2013; Riccio et al. 2025).
            "participation_fraction":     float(participation_fraction),
            # Burst Peak (BP) of MaxLab Live and Axion, here per unit so it is
            # comparable across wells with different yields.
            "burst_peak_hz_per_unit":     float(np.max(population_firing_rate_signal[start_idx:end_idx + 1])),
            # The same peak, array-wide. Rises with unit yield; kept only for
            # comparison against platform software that reports it that way.
            "burst_peak_hz_array":        float(np.max(population_firing_rate_hz[start_idx:end_idx + 1])),
            # Integral of the per-unit population rate: mean spikes per unit.
            "burst_area_spikes_per_unit": float(np.sum(population_firing_rate_signal[start_idx:end_idx + 1]) * bin_size),
            # Spikes per Burst (SPB). Yield-dependent; the _per_unit form is
            # MaxLab Live's "spikes per burst per electrode".
            "spikes_per_burst":           spikes_per_burst,
            "spikes_per_burst_per_unit":  float(spikes_per_burst / max(1, n_units)),
            # Burst spike rate (BSR, Mossink et al. 2021): firing rate of the
            # units that took part, free of both duration and yield.
            "intraburst_rate_hz":         float(burst_density),
            # Mean interval between spikes inside the burst (bISI in MaxLab
            # Live). Falls as a burst becomes internally denser.
            "intraburst_isi_mean_s":      _intraburst_isi_mean(all_spikes, start_time_s, end_time_s),
        })

    # ---------------------------------------------------------
    # 7. Merge logic
    # ---------------------------------------------------------
    def finalize(evs, s, e):

        best = max(evs, key=lambda x: x["coactive_fraction_peak"])

        participating_units = sum(
            1 for u in units
            if np.any((SpikeTimes[u] >= s) & (SpikeTimes[u] < e))
        )

        # Re-integrate over the whole merged window [s, e] instead of summing
        # the components. Summing skipped the gaps between components while
        # burst_duration_s and participation_fraction span them, so
        # spikes_per_burst / burst_duration_s was not the burst's firing rate.
        # The merged window's bounds are component bounds, which are bin
        # edges, so these indices are exact.
        i0 = int(np.clip(np.searchsorted(bins, s, side="left"), 0, n_bins - 1))
        i1 = int(np.clip(np.searchsorted(bins, e, side="left") - 1, i0, n_bins - 1))

        burst_duration_s = e - s
        spikes_per_burst = int(np.sum(spike_counts_total[i0:i1 + 1]))
        denom = burst_duration_s * max(1, participating_units)

        peak_time_s = best["peak_time_s"]
        return {
            "start_time_s":               s,
            "end_time_s":                 e,
            "burst_duration_s":           burst_duration_s,
            "peak_time_s":                peak_time_s,
            # Measured from the merged burst's own onset, so these grow when
            # fragments merge. Compare them within a tier, not across tiers.
            "rise_time_s":                max(0.0, peak_time_s - s),
            "decay_time_s":               max(0.0, e - peak_time_s),
            "coactive_fraction_peak":     best["coactive_fraction_peak"],
            # Highest single-bin co-activity anywhere in the merged window,
            # from the raw counts. Dropped at this tier before schema 3.
            "coactive_fraction_max":      float(np.max(active_unit_counts[i0:i1 + 1]) / max(1, n_units)),
            "participation_fraction":     participating_units / n_units,
            "burst_peak_hz_per_unit":     float(np.max(population_firing_rate_signal[i0:i1 + 1])),
            "burst_peak_hz_array":        float(np.max(population_firing_rate_hz[i0:i1 + 1])),
            "burst_area_spikes_per_unit": float(np.sum(population_firing_rate_signal[i0:i1 + 1]) * bin_size),
            "spikes_per_burst":           spikes_per_burst,
            "spikes_per_burst_per_unit":  float(spikes_per_burst / max(1, n_units)),
            "intraburst_rate_hz":         float(spikes_per_burst / denom) if denom > 0 else 0.0,
            "intraburst_isi_mean_s":      _intraburst_isi_mean(all_spikes, s, e),
            # Fragments contained, transitively: at the superburst tier this
            # counts fragments, while n_components counts the network bursts
            # merged. The old name "component_count" was read as the latter.
            "n_fragments":                sum(ev.get("n_fragments", 1) for ev in evs),
            "n_components":               len(evs),
        }

    def get_valley_min(prev, nxt, coactive_fraction_signal, t_centers):
        valley_mask = (t_centers >= prev["end_time_s"]) & (t_centers <= nxt["start_time_s"])
        if not np.any(valley_mask):
            return None
        valley_vals = coactive_fraction_signal[valley_mask]
        if valley_vals.size == 0:
            return None
        return float(np.min(valley_vals))

    def merge_by_continuity(events, floor_val, max_gap_s=None, min_dur=0,
                            merge_rule="continuity"):
        """Fragment -> network burst merge, on continuity of network activity.

        Two fragments belong to one network burst when the network never fell
        quiet between them. Fragments end where the co-activity signal drops to
        `extent_frac` of their own peak, which is above `floor_val`, so a dip
        that stays above the detection threshold is a within-burst trough and a
        dip that crosses it is a real silence.

        There is deliberately no gap parameter. The criterion is a property of
        the population signal at the level being merged. Earlier versions
        gated on the 95th percentile of within-burst *spike* intervals from
        individual units, which is one to two orders of magnitude shorter than
        the pause between sub-bursts (milliseconds against tens to hundreds of
        milliseconds), so no two fragments were ever close enough to merge and
        this tier did nothing.

        `max_gap_s` is an optional safety cap for the pathological case of a
        well whose baseline sits above the detection threshold for minutes; it
        is None by default and is recorded in the diagnostics when used.
        """
        if not events:
            return []

        events = sorted(events, key=lambda x: x["start_time_s"])

        merged = []
        curr   = [events[0]]
        s      = events[0]["start_time_s"]
        e      = events[0]["end_time_s"]

        for nxt in events[1:]:

            valley_duration = nxt["start_time_s"] - e
            valley_min      = get_valley_min(curr[-1], nxt, coactive_fraction_signal, t_centers)

            if valley_min is None:
                # Fragments touching to within one bin: no trough to measure.
                continuous = (valley_duration <= bin_size)
            else:
                continuous = (valley_min >= floor_val)

            within_cap = (max_gap_s is None) or (valley_duration <= max_gap_s)

            if merge_rule == "none":
                should_merge = False
            elif merge_rule == "gap":
                # Comparison condition: proximity alone, ignoring whether the
                # network fell silent. This is what the published detectors do
                # (Parodi et al. 2026, 2 s; Xue et al. 2022, 75 ms).
                should_merge = within_cap and valley_duration <= (
                    max_gap_s if max_gap_s is not None else GAP_ONLY_DEFAULT_S)
            else:
                should_merge = continuous and within_cap

            if should_merge:
                curr.append(nxt)
                e = max(e, nxt["end_time_s"])
            else:
                merged.append(finalize(curr, s, e))
                curr = [nxt]
                s    = nxt["start_time_s"]
                e    = nxt["end_time_s"]

        merged.append(finalize(curr, s, e))

        return [m for m in merged if m["burst_duration_s"] >= min_dur]

    def merge_superbursts(events, gap, min_dur=2.5, min_components=2):
        """
        Network burst -> superburst merge.

        Superbursts are prolonged episodes of elevated network activity
        containing several network bursts (Wagenaar et al. 2006: duration
        > 2.5s). Detection is gap-only — no valley floor applied because
        superbursts can contain full silences between component NBs.

        Parameters
        ----------
        gap : float
            Maximum inter-NB gap (s) to merge. Derived from inter-fragment
            interval antimode (section 6b).
        min_dur : float
            Minimum superburst duration in seconds. Default 2.5s per
            Wagenaar et al. 2006 operational definition.
        min_components : int
            Minimum number of component NBs. Default 2, so a superburst is a
            *cluster* of network bursts as in Wagenaar et al. 2006. With
            min_components=1 every network burst longer than min_dur is also
            reported as a superburst, which conflates a single prolonged
            burst (typical of organoids) with a superburst train (typical of
            mature dissociated cultures) and double-counts it across tiers.
            Long single bursts are captured by the network burst tier's
            burst_duration_p95_s / burst_duration_max_s instead.
        """
        if not events:
            return []

        events = sorted(events, key=lambda x: x["start_time_s"])

        merged = []
        curr   = [events[0]]
        s      = events[0]["start_time_s"]
        e      = events[0]["end_time_s"]

        for nxt in events[1:]:
            gap_to_next = nxt["start_time_s"] - e
            if gap_to_next <= gap:
                curr.append(nxt)
                e = max(e, nxt["end_time_s"])
            else:
                merged.append(finalize(curr, s, e))
                curr = [nxt]
                s    = nxt["start_time_s"]
                e    = nxt["end_time_s"]

        merged.append(finalize(curr, s, e))

        return [
            m for m in merged
            if m["burst_duration_s"] >= min_dur
            and m["n_components"] >= min_components
        ]

    network_bursts = merge_by_continuity(
        burst_fragments,
        floor_val=detection_threshold,
        max_gap_s=fragment_max_gap_s,
        merge_rule=merge_rule,
    )

    # ---------------------------------------------------------
    # 7b. Superburst gap, from the network bursts' own interval distribution
    #
    # A superburst is a cluster of network bursts, so the timescale that
    # separates "within a cluster" from "between clusters" lives in the
    # network burst inter-burst intervals — not in the inter-fragment
    # intervals of the tier below, which is what earlier versions used.
    #
    # If that distribution has two modes, the antimode between them is the
    # grouping timescale and the data has told us there is cluster structure.
    # If it has one mode, the network bursts are not clustered and there are
    # no superbursts to find; reporting none is the honest answer, and better
    # than a constant fallback that manufactures superbursts in a regularly
    # bursting culture.
    # ---------------------------------------------------------
    nb_starts = np.array(sorted(b["start_time_s"] for b in network_bursts))
    nb_ibis = np.diff(nb_starts) if nb_starts.size > 1 else np.asarray([])
    nb_ibis = nb_ibis[nb_ibis > 0]

    nb_ibi_bimodality = None
    superburst_gap_s = None
    superburst_gap_source = "too_few_network_bursts"

    if nb_ibis.size >= MIN_IBIS_FOR_SUPERBURST_TEST:
        log_ibis = np.log10(nb_ibis)
        n_ibi = log_ibis.size
        if n_ibi >= 4:
            _g1 = skew(log_ibis)
            _g2 = sp_kurtosis(log_ibis, fisher=True)
            nb_ibi_bimodality = float(
                (_g1**2 + 1) / (_g2 + 3 * ((n_ibi - 1)**2 / ((n_ibi - 2) * (n_ibi - 3))))
            )

        split = _otsu_split(log_ibis)

        if (nb_ibi_bimodality is not None
                and nb_ibi_bimodality > BIMODALITY_THRESHOLD
                and split is not None):
            superburst_gap_s = float(10 ** split)
            superburst_gap_source = "nb_ibi_otsu"
        else:
            superburst_gap_source = "nb_ibi_not_bimodal"

    # A user-supplied gap overrides the test, for call-site compatibility.
    if network_merge_gap_min is not None and superburst_gap_s is None:
        superburst_gap_s = float(network_merge_gap_min)
        superburst_gap_source = "user_override"

    if superburst_gap_s is None:
        superbursts = []
    else:
        superbursts = merge_superbursts(
            network_bursts,
            gap=superburst_gap_s,
            min_dur=min_superburst_dur_s,
            min_components=min_superburst_components,
        )

    # ---------------------------------------------------------
    # 8. Metrics
    # ---------------------------------------------------------
    def level_metrics(events, ibi_key="ibi_s"):
        return _level_metrics(events, total_dur, ibi_key=ibi_key)

    # ---------------------------------------------------------
    # 9. Return
    # ---------------------------------------------------------
    return {

        "burst_fragments": {
            "events":  burst_fragments,
            "metrics": level_metrics(burst_fragments, ibi_key="ifbi_s")
        },
        "network_bursts": {
            "events":  network_bursts,
            "metrics": level_metrics(network_bursts, ibi_key="ibi_s")
        },
        "superbursts": {
            "events":  superbursts,
            "metrics": level_metrics(superbursts, ibi_key="isbi_s")
        },

        # Share of the well's spiking that happens inside network bursts.
        # PRS (percent_random_spikes) is its complement, a core Mossink
        # et al. 2021 parameter: a culture can hold its burst rate while
        # its neurons drift out of the bursts, and only this ratio shows it.
        "spike_participation": spike_participation(all_spikes, network_bursts),

        "diagnostics": {
            "schema_version":               SCHEMA_VERSION,
            "detector":                     "parameter_free",
            "recording_duration_s":         total_dur,
            "analysis_window_s":            analysis_window_s,
            "duration_source":              duration_source,
            "bin_size_ms":                  bin_size_ms,
            "reference_isi_s":              reference_isi_s,
            "reference_isi_source":         "bursty_peak" if len(bursty_log_isis) > 50 else ("all_percentile15" if all_log_isis else "default"),
            "coactive_fraction_baseline":       coactive_fraction_baseline,
            "coactive_fraction_mad":            coactive_fraction_mad,
            "coactive_fraction_bimodality":             coactive_fraction_bimodality,
            "burst_detection_valid":        True,
            "threshold_source":             threshold_source,
            "detection_threshold":          detection_threshold,
            "min_coactive_fraction":        min_coactive_fraction,
            "min_units_for_burst":          min_units_for_burst,
            # Fragment -> network burst: continuity of the population signal,
            # no gap threshold. The cap is None unless a caller sets one.
            "fragment_merge_rule":          merge_rule,
            "fragment_max_gap_s":           fragment_max_gap_s,
            # Network burst -> superburst: from the network bursts' own IBIs.
            "nb_ibi_bimodality":            nb_ibi_bimodality,
            "superburst_gap_s":             superburst_gap_s,
            "superburst_gap_source":        superburst_gap_source,
            "superburst_min_dur_s":         min_superburst_dur_s,
            "superburst_min_components":    min_superburst_components,
            "n_units":                      n_units,
            "n_bursty_units_by_isi_statistics":               sum(1 for s in unit_stats.values() if s.get("is_bursty_by_isi_statistics")),
            "sigma_coactivity_bins":     sigma_coactivity_bins,
            "sigma_firing_rate_bins":       sigma_firing_rate_bins,
        },

        "unit_stats": unit_stats,

        # plot_data is splatted into helper.plot_clean_network as **kwargs, so
        # these keys must match its signature exactly — no aliases. The trace
        # named coactive_fraction_signal is per-bin co-activity
        # (synchrony), not the per-burst participation_fraction, and
        # population_firing_rate_hz here is per unit. See docs/metrics.md.
        "plot_data": {
            "time_s":                          t_centers,
            "coactive_fraction_signal":   coactive_fraction_signal,
            "population_firing_rate_hz":       population_firing_rate_signal,
            "nb_peak_times_s":                 np.array([b["peak_time_s"] for b in network_bursts]),
            "nb_peak_coactive_fraction":  np.array([b["coactive_fraction_peak"] for b in network_bursts]),
            "sb_start_times_s":                np.array([b["start_time_s"] for b in superbursts]),
            "sb_end_times_s":                  np.array([b["end_time_s"] for b in superbursts]),
            "coactive_fraction_baseline":          coactive_fraction_baseline,
            "detection_threshold":             detection_threshold,
        }
    }