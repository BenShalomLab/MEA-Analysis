# ==========================================================
# catch22_features.py
# Author: Mandar Patil
# LLM Assisted Edits: Yes Claude Sonnet 5
#
# Thin wrapper around pycatch22 (github.com/DynamicsAndNeuralSystems/catch22)
# used at two levels in this pipeline:
#   - per-unit: one binned spike train -> 22 features (feeds UnitMatch /
#     unit clustering / phenotyping).
#   - population: the smoothed population firing-rate trace already built
#     by gaussianNetworkBursts.compute_network_bursts -> 22 features
#     characterizing whole-recording network dynamics.
#
# Both call sites reuse compute_catch22() below so feature names/behavior
# (esp. the "too short to compute" fallback) stay identical everywhere,
# mirroring the burst_common.stats() convention already used in this repo.
# ==========================================================

import numpy as np

try:
    import pycatch22
except ImportError:
    pycatch22 = None

# catch22 features need enough samples to estimate autocorrelation/entropy
# measures; shorter series produce garbage or raise inside the C backend.
MIN_LENGTH = 20


def compute_catch22(x, catch24=False):
    """Run catch22 on a 1D array-like, returning {feature_name: float}.

    Mirrors burst_common.stats()'s convention of returning a fixed-shape
    dict of NaNs instead of raising, so callers/JSON/Excel consumers never
    have to special-case a missing unit or an empty/short recording.

    Parameters
    ----------
    x : array-like
        1D time series (e.g. one unit's binned spike train, or the
        population firing-rate trace).
    catch24 : bool
        Also include the 2 extra catch24 features (mean, spread) on top of
        the 22 catch22 features. Off by default since this pipeline's own
        stats() already reports mean/std/cv separately.
    """
    if pycatch22 is None:
        raise ImportError(
            "pycatch22 is not installed. Install it with `pip install pycatch22`."
        )

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < MIN_LENGTH:
        names = pycatch22.catch22_all([0.0] * MIN_LENGTH, catch24=catch24)["names"]
        return {name: float("nan") for name in names}

    result = pycatch22.catch22_all(x.tolist(), catch24=catch24)
    return dict(zip(result["names"], (float(v) for v in result["values"])))


def compute_unit_catch22_features(SpikeTimes, bin_size_s, t_start, t_end):
    """Per-unit catch22 features from binned spike trains.

    Parameters
    ----------
    SpikeTimes : dict[unit_id, array-like]
        Spike times in seconds, per unit (same shape as everywhere else in
        this pipeline, e.g. gaussianNetworkBursts.compute_network_bursts).
    bin_size_s : float
        Histogram bin width (s). Use the same value the caller used to
        build its population-rate trace so unit- and population-level
        features are computed on directly comparable timescales.
    t_start, t_end : float
        Recording window (s) to bin over — pass the same window used
        elsewhere (e.g. rec_start/rec_end) so all units share bin edges.

    Returns
    -------
    dict[unit_id, dict[str, float]]
    """
    if t_end <= t_start:
        return {u: compute_catch22([]) for u in SpikeTimes}

    bins = np.arange(t_start, t_end + bin_size_s, bin_size_s)

    unit_features = {}
    for u, spikes in SpikeTimes.items():
        spikes = np.asarray(spikes)
        counts, _ = np.histogram(spikes, bins=bins) if spikes.size else (np.array([]), None)
        unit_features[u] = compute_catch22(counts)

    return unit_features
