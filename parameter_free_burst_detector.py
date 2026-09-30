import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis as sp_kurtosis

try:
    from burst_common import stats, level_metrics as _level_metrics, SCHEMA_VERSION
except ImportError:
    from MEA_Analysis.IPNAnalysis.burst_common import stats, level_metrics as _level_metrics, SCHEMA_VERSION


def compute_network_bursts(
    SpikeTimes=None,
    duration_s=None,
    extent_frac=0.30,
    network_merge_gap_min=0.75,
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
        cv_isi  = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else np.nan

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

        is_bursty = bool((not np.isnan(bc)) and bc > 0.555 and (np.isnan(lv) or lv > 1.0))

        unit_stats[u] = {
            "mean_firing_rate_hz":    mean_fr,
            "cv_isi":                 cv_isi,
            "cv2":                    cv2,
            "lv":                     lv,
            "bimodality_coefficient": float(bc) if not np.isnan(bc) else None,
            "is_bursty":              is_bursty,
        }

        if is_bursty:
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

    participation_fraction_signal_raw = active_unit_counts / max(1, n_units)
    rate_signal_raw                   = spike_counts_total / bin_size / max(1, n_units)

    population_firing_rate_hz = spike_counts_total / bin_size

    # ---------------------------------------------------------
    # 3. Smoothing
    # ---------------------------------------------------------
    isi_bins = reference_isi_s / bin_size

    sigma_participation_bins = np.clip(isi_bins, 1, 2)
    sigma_firing_rate_bins   = np.clip(5.0 * isi_bins, 3, 8)

    participation_fraction_signal = gaussian_filter1d(participation_fraction_signal_raw, sigma_participation_bins)
    population_firing_rate_signal = gaussian_filter1d(rate_signal_raw, sigma_firing_rate_bins)

    # ---------------------------------------------------------
    # 3b. Adaptive merge gaps
    #
    # fragment_merge_gap_s: 95th percentile of intra-burst ISIs from bursty
    #   units — empirical ceiling of within-burst pauses (Bakkum et al. 2013).
    #   Population log-ISI antimode NOT used — at high firing rates the
    #   distribution is unimodal and antimode finds sub-ms refractory
    #   artifacts rather than the intra/inter-burst boundary.
    #
    # nb_merge_gap_s: anti-mode of inter-fragment interval distribution,
    #   computed after fragment extraction (section 6b). Falls back to STD
    #   recovery floor of 0.3s (Tsodyks & Markram 1997: 300-1500ms).
    # ---------------------------------------------------------

    intra_burst_isis = []
    for u in units:
        if not unit_stats[u].get("is_bursty"):
            continue
        t = np.unique(np.sort(SpikeTimes[u]))
        isi = np.diff(t)
        isi = isi[isi > 0]
        if len(isi) < 10:
            continue
        log_isi = np.log10(isi)
        h, e = np.histogram(log_isi, bins=50)
        c    = (e[:-1] + e[1:]) / 2
        hs   = gaussian_filter1d(h.astype(float), sigma=2)
        v, _ = find_peaks(-hs, prominence=1)
        if len(v) > 0:
            antimode_s = float(10 ** c[v[0]])
            intra_burst_isis.extend(isi[isi < antimode_s].tolist())

    if len(intra_burst_isis) > 20:
        fragment_merge_gap_s      = float(np.percentile(intra_burst_isis, 95))
        fragment_merge_gap_source = "intra_burst_isi_p95"
    else:
        fragment_merge_gap_s      = 3 * reference_isi_s
        fragment_merge_gap_source = "fallback_3x_isi"

    # ---------------------------------------------------------
    # 4. Detection thresholds
    # ---------------------------------------------------------
    participation_baseline = np.median(participation_fraction_signal)
    participation_mad      = np.median(np.abs(participation_fraction_signal - participation_baseline))

    # Bimodality coefficient on participation signal (Sarle 1990).
    # BC selects threshold method — NOT used as a hard gate.
    # Sparse-burst recordings produce low BC even with genuine burst
    # structure because the burst population is too small relative to
    # the baseline to create a visible second mode. Hard gating on BC
    # causes false negatives in exactly these cases.
    _pf = participation_fraction_signal
    _n  = len(_pf)
    if _n >= 4:
        _g1 = skew(_pf)
        _g2 = sp_kurtosis(_pf, fisher=True)
        participation_bc = float(
            (_g1**2 + 1) / (_g2 + 3 * ((_n - 1)**2 / ((_n - 2) * (_n - 3))))
        )
    else:
        participation_bc = 0.0

    if participation_bc > 0.555:
        # Signal is genuinely bimodal — MAD robustly captures the
        # burst/baseline separation.
        detection_threshold = max(
            0.03,
            participation_baseline + threshold_mad_scale * participation_mad
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
            float(np.percentile(participation_fraction_signal, pct))
        )
        threshold_source = f"p{pct:.1f}"

    # Prominence: peaks must rise sharply above local surroundings.
    # 2*MAD filters broad low-amplitude elevations with no synchrony structure.
    min_prominence = max(2.0 * participation_mad, 0.02)

    # Adaptive peak synchrony floor:
    # min_peak_synchrony as passed is interpreted as a fraction, but for
    # small cultures a single unit firing creates a 1/n_units jump that
    # trivially clears any fixed fraction floor.
    # Require at least max(3, 10% of n_units) units co-active in a single
    # bin — scales the absolute unit count floor with culture size.
    min_units_for_burst      = max(3, int(0.10 * n_units))
    min_peak_synchrony_adaptive = max(
        min_peak_synchrony,
        min_units_for_burst / max(1, n_units)
    )

    # ---------------------------------------------------------
    # 5. Peak detection
    # ---------------------------------------------------------
    peaks, _ = find_peaks(
        participation_fraction_signal,
        height=detection_threshold,
        prominence=min_prominence,
    )

    burst_fragments = []

    # ---------------------------------------------------------
    # 6. Fragment extraction
    # ---------------------------------------------------------
    for p in peaks:

        peak_val         = participation_fraction_signal[p]
        extent_threshold = max(detection_threshold, extent_frac * peak_val)

        # LEFT boundary
        s = p
        while s > 0 and participation_fraction_signal[s - 1] >= extent_threshold:
            s -= 1

        # RIGHT boundary
        e = p
        while e < n_bins - 1 and participation_fraction_signal[e + 1] >= extent_threshold:
            e += 1

        start_idx = s
        end_idx   = e

        start_time_s = bins[start_idx]
        end_time_s   = bins[end_idx + 1]

        burst_duration_s = end_time_s - start_time_s
        if burst_duration_s <= 0:
            continue

        # Peak synchrony validation: require min_peak_synchrony_adaptive
        # fraction of units co-active in a single bin within the fragment.
        # Adaptive floor prevents single-unit noise events from passing in
        # small cultures (n_units < 50) where 1/n_units jumps are large.
        peak_bin_synchrony = float(
            np.max(active_unit_counts[start_idx:end_idx + 1]) / max(1, n_units)
        )
        if peak_bin_synchrony < min_peak_synchrony_adaptive:
            continue

        participating = sum(
            1 for u in units
            if np.any((SpikeTimes[u] >= start_time_s) & (SpikeTimes[u] < end_time_s))
        )

        participation_fraction = participating / n_units

        if min_fragment_participation > 0 and participation_fraction < min_fragment_participation:
            continue

        spike_count = int(np.sum(spike_counts_total[start_idx:end_idx + 1]))

        denom         = burst_duration_s * max(1, participating)
        burst_density = spike_count / denom if denom > 0 else 0

        peak_drive_rate = np.max(rate_signal_raw[start_idx:end_idx + 1])

        if min_burst_density_Hz > 0 and burst_density < min_burst_density_Hz:
            continue

        if min_absolute_rate_Hz > 0 and peak_drive_rate < min_absolute_rate_Hz:
            continue

        burst_fragments.append({
            "start_time_s":                   float(start_time_s),
            "end_time_s":                     float(end_time_s),
            "burst_duration_s":               float(burst_duration_s),
            # Per-bin co-activity at the peak. "peak_synchrony" is the accurate
            # name; "peak_participation_fraction" is kept because it is the key
            # existing notebooks and collected tables read. Neither is the
            # per-burst `participation_fraction` below — see docs/metrics.md.
            "peak_synchrony":                 float(peak_val),
            "peak_participation_fraction":    float(peak_val),
            "peak_time_s":                    float(t_centers[p]),
            # Integral of the per-unit population rate: mean spikes per unit.
            "burst_area":                     float(np.sum(population_firing_rate_signal[start_idx:end_idx + 1]) * bin_size),
            "participation_fraction":         float(participation_fraction),
            "spike_count":                    spike_count,
            # Yield-normalised counterpart of spike_count, so wells with
            # different unit yields are comparable.
            "spikes_per_burst_per_unit":      float(spike_count / max(1, n_units)),
            # Spikes per participating unit per second inside the burst. Used
            # as a detection gate above and reported because it is the
            # intensity measure that is free of both duration and yield.
            "burst_density_hz":               float(burst_density),
            "peak_bin_synchrony":             peak_bin_synchrony,
            # Per unit, from the same smoothed signal as burst_area, so every
            # reported intensity is in the same convention as MaxWell's
            # mxw.networkActivity.computeNetworkAct and as the Gaussian
            # detector's field of this name.
            "peak_population_firing_rate_hz": float(np.max(population_firing_rate_signal[start_idx:end_idx + 1])),
            # Array-wide sum, NOT divided by n_units. This is what the field
            # above used to hold; it is kept under an explicit name because it
            # rises with unit yield and must not be compared across wells.
            "peak_population_firing_rate_total_hz": float(np.max(population_firing_rate_hz[start_idx:end_idx + 1])),
        })

    # ---------------------------------------------------------
    # 6b. nb_merge_gap_s — derived from inter-fragment interval distribution
    #
    # Anti-mode of log(inter-fragment intervals) separates short within-
    # superburst gaps (~1s, driven by STD/facilitation cycling) from long
    # true IBIs (tens of seconds, driven by Nap current recharge and AHP).
    # Floor of 0.3s = low end of cortical vesicle recovery range
    # (Tsodyks & Markram 1997). network_merge_gap_min preserved as
    # user-overridable floor for call-site compatibility.
    # ---------------------------------------------------------
    if len(burst_fragments) > 3:
        _frag_starts = np.array(sorted(f["start_time_s"] for f in burst_fragments))
        _ifis        = np.diff(_frag_starts)
        _ifis        = _ifis[_ifis > 0]
        if len(_ifis) > 5:
            _log_ifis         = np.log10(_ifis)
            _hist_i, _edges_i = np.histogram(_log_ifis, bins=min(50, len(_log_ifis) // 2))
            _centers_i        = (_edges_i[:-1] + _edges_i[1:]) / 2
            _smooth_i         = gaussian_filter1d(_hist_i.astype(float), sigma=2)
            _valleys_i, _     = find_peaks(-_smooth_i, prominence=2)
            if len(_valleys_i) > 0:
                nb_merge_gap_s      = float(10 ** _centers_i[_valleys_i[0]])
                nb_merge_gap_source = "inter_fragment_antimode"
            else:
                nb_merge_gap_s      = max(network_merge_gap_min, 0.3)
                nb_merge_gap_source = "fallback_floor"
        else:
            nb_merge_gap_s      = max(network_merge_gap_min, 0.3)
            nb_merge_gap_source = "fallback_floor"
    else:
        nb_merge_gap_s      = max(network_merge_gap_min, 0.3)
        nb_merge_gap_source = "fallback_floor"

    # ---------------------------------------------------------
    # 7. Merge logic
    # ---------------------------------------------------------
    def finalize(evs, s, e):

        best = max(evs, key=lambda x: x["peak_synchrony"])

        participating_units = sum(
            1 for u in units
            if np.any((SpikeTimes[u] >= s) & (SpikeTimes[u] < e))
        )

        # Re-integrate over the whole merged window [s, e] instead of summing
        # the components. Summing skipped the gaps between components while
        # burst_duration_s and participation_fraction span them, so
        # spike_count / burst_duration_s was not the burst's firing rate.
        # The merged window's bounds are component bounds, which are bin
        # edges, so these indices are exact.
        i0 = int(np.clip(np.searchsorted(bins, s, side="left"), 0, n_bins - 1))
        i1 = int(np.clip(np.searchsorted(bins, e, side="left") - 1, i0, n_bins - 1))

        burst_duration_s = e - s
        spike_count = int(np.sum(spike_counts_total[i0:i1 + 1]))
        denom = burst_duration_s * max(1, participating_units)

        return {
            "start_time_s":                   s,
            "end_time_s":                     e,
            "burst_duration_s":               burst_duration_s,
            "peak_synchrony":                 best["peak_synchrony"],
            "peak_participation_fraction":    best["peak_synchrony"],
            "peak_time_s":                    best["peak_time_s"],
            "burst_area":                     float(np.sum(population_firing_rate_signal[i0:i1 + 1]) * bin_size),
            # Fragments contained, transitively: at the superburst tier this
            # counts fragments, while n_components counts the network bursts
            # merged. The old name "component_count" was read as the latter.
            "n_fragments":                    sum(ev.get("n_fragments", 1) for ev in evs),
            "spike_count":                    spike_count,
            "spikes_per_burst_per_unit":      float(spike_count / max(1, n_units)),
            "burst_density_hz":               float(spike_count / denom) if denom > 0 else 0.0,
            "participation_fraction":         participating_units / n_units,
            # Highest single-bin co-activity anywhere in the merged window,
            # from the raw counts. Previously dropped at this tier.
            "peak_bin_synchrony":             float(np.max(active_unit_counts[i0:i1 + 1]) / max(1, n_units)),
            "peak_population_firing_rate_hz": float(np.max(population_firing_rate_signal[i0:i1 + 1])),
            "peak_population_firing_rate_total_hz": float(np.max(population_firing_rate_hz[i0:i1 + 1])),
            "n_components":                   len(evs)
        }

    def get_valley_min(prev, nxt, participation_fraction_signal, t_centers):
        valley_mask = (t_centers >= prev["end_time_s"]) & (t_centers <= nxt["start_time_s"])
        if not np.any(valley_mask):
            return None
        valley_vals = participation_fraction_signal[valley_mask]
        if valley_vals.size == 0:
            return None
        return float(np.min(valley_vals))

    def merge_strict(events, gap, floor_val, min_dur=0):
        """
        Fragment -> network burst merge.
        Valley floor gates merging: valley must stay above floor_val,
        meaning activity never fully ceased between fragments.
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
            valley_min      = get_valley_min(curr[-1], nxt, participation_fraction_signal, t_centers)

            if valley_min is None:
                valley_ok = (valley_duration <= bin_size)
            else:
                valley_ok = (valley_min >= floor_val)

            merge_condition = (valley_duration <= gap) and valley_ok

            if merge_condition:
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

    network_bursts = merge_strict(
        burst_fragments,
        fragment_merge_gap_s,
        detection_threshold
    )

    superbursts = merge_superbursts(
        network_bursts,
        gap=nb_merge_gap_s,
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

        "diagnostics": {
            "schema_version":               SCHEMA_VERSION,
            "detector":                     "parameter_free",
            "recording_duration_s":         total_dur,
            "analysis_window_s":            analysis_window_s,
            "duration_source":              duration_source,
            "bin_size_ms":                  bin_size_ms,
            "reference_isi_s":              reference_isi_s,
            "reference_isi_source":         "bursty_peak" if len(bursty_log_isis) > 50 else ("all_percentile15" if all_log_isis else "default"),
            "participation_baseline":       participation_baseline,
            "participation_mad":            participation_mad,
            "participation_bc":             participation_bc,
            "burst_detection_valid":        True,
            "threshold_source":             threshold_source,
            "detection_threshold":          detection_threshold,
            "min_peak_synchrony_adaptive":  min_peak_synchrony_adaptive,
            "min_units_for_burst":          min_units_for_burst,
            "fragment_merge_gap_s":         fragment_merge_gap_s,
            "fragment_merge_gap_source":    fragment_merge_gap_source,
            "nb_merge_gap_s":               nb_merge_gap_s,
            "nb_merge_gap_source":          nb_merge_gap_source,
            "superburst_min_dur_s":         min_superburst_dur_s,
            "superburst_min_components":    min_superburst_components,
            "superburst_merge_gap_s":       nb_merge_gap_s,
            "n_units":                      n_units,
            "n_bursty_units":               sum(1 for s in unit_stats.values() if s.get("is_bursty")),
            "sigma_participation_bins":     sigma_participation_bins,
            "sigma_firing_rate_bins":       sigma_firing_rate_bins,
        },

        "unit_stats": unit_stats,

        # plot_data is splatted into helper.plot_clean_network as **kwargs, so
        # these keys must match its signature exactly — no aliases. The trace
        # named participation_fraction_signal is per-bin co-activity
        # (synchrony), not the per-burst participation_fraction, and
        # population_firing_rate_hz here is per unit. See docs/metrics.md.
        "plot_data": {
            "time_s":                          t_centers,
            "participation_fraction_signal":   participation_fraction_signal,
            "population_firing_rate_hz":       population_firing_rate_signal,
            "nb_peak_times_s":                 np.array([b["peak_time_s"] for b in network_bursts]),
            "nb_peak_participation_fraction":  np.array([b["peak_synchrony"] for b in network_bursts]),
            "sb_start_times_s":                np.array([b["start_time_s"] for b in superbursts]),
            "sb_end_times_s":                  np.array([b["end_time_s"] for b in superbursts]),
            "participation_baseline":          participation_baseline,
            "detection_threshold":             detection_threshold,
        }
    }