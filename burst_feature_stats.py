"""
burst_feature_stats.py

Shared distribution-statistics taxonomy for burst-metric feature extraction.

Single source of truth for both feature-extraction notebooks
(ml_model_traditaional.ipynb, Gaussian-based tier, and
ml_model_pythonV2_spikesorted.ipynb, adaptive/burst-detection tier), so the
two feature sets use an identical statistical taxonomy.

Core taxonomy, per burst-metric list (duration, peak, spikes per burst, IBI):
    median   : location. Median is invariant to monotone transforms, so this
               is also the median on a log scale.
    log_iqr  : spread. IQR of natural-log values (positive values only).
               exp(log_iqr) is roughly the ratio between the 75th and 25th
               percentile, so it is scale free and robust to the heavy right
               tail these metrics have.
    p90      : upper tail. 90th percentile, captures prolonged or very large
               events that the median hides.

Sample size n is returned in the dict but is NOT a feature (it is not in
KEEP_STATS). It is used here only to gate statistics that are unreliable at
small n. Burst count enters the model once, as a scalar (Number_Bursts or a
rate), not once per metric.

Optional stats for ablation, requested through extra_stats:
    cv       : SD / mean on the raw scale (regularity, mainly useful for IBI).
    bowley   : quartile skewness on log values, robust replacement for sample
               skew.
    entropy  : Shannon entropy on log values using FIXED bin edges shared by
               all recordings, normalised by log2(n_bins) to lie in [0, 1].
               Requires entropy_edges. Do not use data-dependent bins, they
               make entropy a function of n.

Undefined statistics (too few events) are set to fill_value. Default is NaN so
that "undefined" stays distinct from a real zero. XGBoost and LightGBM handle
NaN natively. For the MLP, impute inside each training fold, or pass
fill_value=0.0 to reproduce the earlier zero-fill behaviour.

Cross-metric redundancy (duration, peak and spikes per burst are strongly
correlated) is NOT handled here. Prune correlated features downstream,
label-free, before modelling.
"""

import numpy as np

# Features that become columns. n is deliberately excluded.
KEEP_STATS = ["median", "log_iqr", "p90"]

# Kept for import compatibility with the notebooks.
DISPERSION_STATS = {"log_iqr", "cv"}
SHAPE_STATS = {"p90", "entropy", "bowley"}

OPTIONAL_STATS = {"cv", "bowley", "entropy"}


def make_log_edges(pooled_values, n_bins=10, q_low=1, q_high=99):
    """
    Build fixed log-scale bin edges for one metric from values pooled over the
    whole dataset. Call once per metric and pass the result as entropy_edges.
    For a strict train/test separation, build edges from training folds only,
    or from a fixed physiological range rather than from the data.
    """
    v = np.asarray(pooled_values, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.percentile(np.log(v), [q_low, q_high])
    return np.linspace(lo, hi, n_bins + 1)


def _fixed_bin_entropy(log_x, edges):
    edges = np.asarray(edges, dtype=float)
    clipped = np.clip(log_x, edges[0], edges[-1])
    hist, _ = np.histogram(clipped, bins=edges)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return np.nan
    p = hist / total
    p = p[p > 0]
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(len(edges) - 1))


def compute_distribution_stats(
    x,
    min_n_spread=3,
    min_n_tail=5,
    fill_value=np.nan,
    extra_stats=(),
    entropy_edges=None,
):
    """
    Compute the compact taxonomy for one recording-level burst-metric array.

    Returns a dict with "n" plus median, log_iqr, p90 and any extra_stats.

    Reliability gating:
      - n == 0                 : every statistic -> fill_value
      - median                 : needs n >= 1
      - log_iqr, cv            : need n >= min_n_spread
      - p90, bowley, entropy   : need n >= min_n_tail
    log_iqr, bowley and entropy use only strictly positive values, and the
    same n thresholds are applied to that positive subset.
    """
    extra_stats = tuple(extra_stats)
    unknown = set(extra_stats) - OPTIONAL_STATS
    if unknown:
        raise ValueError(f"Unknown extra_stats: {sorted(unknown)}")
    if "entropy" in extra_stats and entropy_edges is None:
        raise ValueError("entropy requires entropy_edges (see make_log_edges)")

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)

    names = list(KEEP_STATS) + list(extra_stats)
    out = {k: fill_value for k in names}
    out["n"] = n
    if n == 0:
        return out

    out["median"] = float(np.median(x))

    pos = x[x > 0]
    log_x = np.log(pos)
    n_pos = int(pos.size)

    if n_pos >= min_n_spread:
        q25, q75 = np.percentile(log_x, [25, 75])
        out["log_iqr"] = float(q75 - q25)

    if n >= min_n_tail:
        out["p90"] = float(np.percentile(x, 90))

    if "cv" in extra_stats and n >= min_n_spread:
        m = float(np.mean(x))
        if m > 0:
            out["cv"] = float(np.std(x, ddof=1) / m)

    if n_pos >= min_n_tail:
        if "bowley" in extra_stats:
            q25, q50, q75 = np.percentile(log_x, [25, 50, 75])
            if q75 > q25:
                out["bowley"] = float((q75 + q25 - 2 * q50) / (q75 - q25))
        if "entropy" in extra_stats:
            out["entropy"] = _fixed_bin_entropy(log_x, entropy_edges)

    return out
