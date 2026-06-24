import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis as sp_kurtosis


def compute_network_bursts(
    SpikeTimes=None,
    extent_frac=0.30,
    network_merge_gap_min=0.75,
    threshold_mad_scale=0.75,
    min_fragment_participation=0.0,
    min_burst_density_Hz=0.0,
    min_absolute_rate_Hz=0.0,
    min_superburst_dur_s=2.5,
    min_superburst_components=1,
):

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
    total_dur = rec_end - rec_start

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
            reference_isi_s = float(10 ** centers[peaks[0]])   # short-mode peak
        else:
            reference_isi_s = float(10 ** np.percentile(bursty_log_isis, 15))
    elif all_log_isis:
        # Fallback: no bursty units detected (young / sparse culture)
        reference_isi_s = float(10 ** np.percentile(all_log_isis, 15))
    else:
        reference_isi_s = 0.05

    bin_size_ms = np.clip(reference_isi_s * 1000, 10, 30)
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
    # 3b. Adaptive merge gaps — derived from ISI/IFI distributions
    #
    # fragment_merge_gap_s: anti-mode of the log-ISI distribution separates
    #   intra-burst ISIs from inter-burst ISIs (Wagenaar 2006, Selinger 2007).
    #   Restricted to the short-ISI half of the distribution to avoid picking
    #   up the inter-burst valley. Falls back to 3 * reference_isi_s.
    #
    # nb_merge_gap_s: anti-mode of the inter-fragment interval distribution
    #   separates within-superburst gaps from true IBIs. The 0.3s floor is the
    #   low end of the cortical STD vesicle recovery range (Tsodyks & Markram
    #   1997: 300-1500ms). The 10x multiplier is removed — it had no empirical
    #   basis and was effectively overridden by network_merge_gap_min anyway.
    # ---------------------------------------------------------

    # fragment_merge_gap_s
    if len(all_log_isis) > 20:
        _hist, _edges = np.histogram(
            all_log_isis,
            bins=min(100, len(all_log_isis) // 5)
        )
        _centers    = (_edges[:-1] + _edges[1:]) / 2
        _smooth     = gaussian_filter1d(_hist.astype(float), sigma=2)
        _valleys, _ = find_peaks(-_smooth, prominence=2)
        # restrict to left (short-ISI) half so we find the intra-burst ceiling
        _short_mask    = _centers < np.percentile(all_log_isis, 50)
        _valleys_short = _valleys[_short_mask[_valleys]] if len(_valleys) > 0 else np.array([], dtype=int)
        if len(_valleys_short) > 0:
            fragment_merge_gap_s      = float(10 ** _centers[_valleys_short[0]])
            fragment_merge_gap_source = "log_isi_antimode"
        else:
            fragment_merge_gap_s      = 3 * reference_isi_s
            fragment_merge_gap_source = "fallback_3x_isi"
    else:
        fragment_merge_gap_s      = 3 * reference_isi_s
        fragment_merge_gap_source = "fallback_3x_isi"

    # nb_merge_gap_s — computed after burst_fragments is populated below,
    # so we defer it to section 6b.

    # ---------------------------------------------------------
    # 4. Detection thresholds
    # ---------------------------------------------------------
    participation_floor_count = max(5, 0.15 * n_units) if n_units < 50 else max(10, 0.05 * n_units)
    participation_floor       = participation_floor_count / max(1, n_units)

    participation_baseline = np.median(participation_fraction_signal)
    participation_mad      = np.median(np.abs(participation_fraction_signal - participation_baseline))

    detection_threshold = max(participation_floor, participation_baseline + threshold_mad_scale * participation_mad)

    # ---------------------------------------------------------
    # 5. Peak detection
    # ---------------------------------------------------------
    min_prominence = max(0.5 * participation_mad, 0.02)

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
            "peak_participation_fraction":    float(peak_val),
            "peak_time_s":                    float(t_centers[p]),
            "burst_area":                     float(np.sum(population_firing_rate_signal[start_idx:end_idx + 1]) * bin_size),
            "participation_fraction":         float(participation_fraction),
            "spike_count":                    spike_count,
            "peak_population_firing_rate_hz": float(np.max(population_firing_rate_hz[start_idx:end_idx + 1]))
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

        best = max(evs, key=lambda x: x["peak_participation_fraction"])

        participating_units = sum(
            1 for u in units
            if np.any((SpikeTimes[u] >= s) & (SpikeTimes[u] < e))
        )

        return {
            "start_time_s":                   s,
            "end_time_s":                     e,
            "burst_duration_s":               e - s,
            "peak_participation_fraction":    best["peak_participation_fraction"],
            "peak_time_s":                    best["peak_time_s"],
            "burst_area":                     sum(ev["burst_area"] for ev in evs),
            "component_count":                sum(ev.get("component_count", 1) for ev in evs),
            "spike_count":                    sum(ev["spike_count"] for ev in evs),
            "participation_fraction":         participating_units / n_units,
            "peak_population_firing_rate_hz": max(ev["peak_population_firing_rate_hz"] for ev in evs),
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

    def merge_superbursts(events, gap, min_dur=2.5, min_components=1):
        """
        Network burst -> superburst merge.

        Superbursts are prolonged episodes of elevated network activity
        containing one or more network bursts (Wagenaar et al. 2006:
        duration > 2.5s). Detection is gap-only — no valley floor is
        applied because superbursts can contain full silences between
        component NBs (valley floor would pathologically split them).

        Parameters
        ----------
        gap : float
            Maximum inter-NB gap (s) to merge into a superburst.
            Derived from inter-fragment interval antimode (section 6b).
        min_dur : float
            Minimum superburst duration in seconds. Default 2.5s per
            Wagenaar et al. 2006 operational definition.
        min_components : int
            Minimum number of component NBs. Default 1 to include long
            single NBs that represent sustained reverberant recruitment.
            Set to 2 to require explicit multi-NB clustering.
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
    def stats(x):

        x = np.asarray(x)

        if x.size == 0:
            return {"mean": 0.0, "std": 0.0, "cv": 0.0}

        mean_val = x.mean()
        std_val  = x.std()
        cv       = std_val / mean_val if abs(mean_val) > 1e-12 else np.nan

        return {
            "mean": float(mean_val),
            "std":  float(std_val),
            "cv":   float(cv)
        }

    def level_metrics(events, ibi_key="ibi_s"):

        if not events:
            return {}

        starts = [ev["start_time_s"] for ev in events]

        return {
            "burst_count":                    len(events),
            "burst_rate_hz":                  len(events) / total_dur,
            "burst_duration_s":               stats([ev["burst_duration_s"] for ev in events]),
            ibi_key:                          stats(np.diff(starts)) if len(starts) > 1 else stats([]),
            "burst_area":                     stats([ev["burst_area"] for ev in events]),
            "participation_fraction":         stats([ev["participation_fraction"] for ev in events]),
            "spike_count_per_burst":          stats([ev["spike_count"] for ev in events]),
            "peak_population_firing_rate_hz": stats([ev["peak_population_firing_rate_hz"] for ev in events]),
            "peak_participation_fraction":    stats([ev["peak_participation_fraction"] for ev in events]),
        }

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
            "bin_size_ms":               bin_size_ms,
            "reference_isi_s":           reference_isi_s,
            "reference_isi_source":      "bursty_peak" if len(bursty_log_isis) > 50 else ("all_percentile15" if all_log_isis else "default"),
            "participation_baseline":    participation_baseline,
            "participation_mad":         participation_mad,
            "detection_threshold":       detection_threshold,
            "fragment_merge_gap_s":      fragment_merge_gap_s,
            "fragment_merge_gap_source": fragment_merge_gap_source,
            "nb_merge_gap_s":            nb_merge_gap_s,
            "nb_merge_gap_source":       nb_merge_gap_source,
            "superburst_min_dur_s":      min_superburst_dur_s,
            "superburst_merge_gap_s":    nb_merge_gap_s,
            "n_units":                   n_units,
            "n_bursty_units":            sum(1 for s in unit_stats.values() if s.get("is_bursty")),
            "sigma_participation_bins":  sigma_participation_bins,
            "sigma_firing_rate_bins":    sigma_firing_rate_bins,
        },

        "unit_stats": unit_stats,

        "plot_data": {
            "time_s":                          t_centers,
            "participation_fraction_signal":   participation_fraction_signal,
            "population_firing_rate_hz":       population_firing_rate_signal,
            "nb_peak_times_s":                 np.array([b["peak_time_s"] for b in network_bursts]),
            "nb_peak_participation_fraction":  np.array([b["peak_participation_fraction"] for b in network_bursts]),
            "sb_start_times_s":                np.array([b["start_time_s"] for b in superbursts]),
            "sb_end_times_s":                  np.array([b["end_time_s"] for b in superbursts]),
            "participation_baseline":          participation_baseline,
            "detection_threshold":             detection_threshold,
        }
    }