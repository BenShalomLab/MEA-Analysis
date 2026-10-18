"""Bad-channel detection from per-channel noise and amplitude.

Pure numpy so it can be tested without SpikeInterface. A channel is bad when its
noise or its amplitude is a robust outlier among the array's channels, on a log
scale (both quantities are positive and roughly log-normal across a chip).
"""
import numpy as np

# Robust z-score (median/MAD of log values) beyond which a channel is an outlier.
Z_THRESHOLD = 5.0
# Amplitude statistic: high percentile of |x - median|. Percentile (not max) so a
# single transient does not decide it.
AMPLITUDE_PERCENTILE = 99.9


def channel_noise_amplitude(data):
    """data: (n_samples, n_channels). Returns (noise, amplitude), each per channel.

    noise     = MAD / 0.6745  (robust std, ignores spikes)
    amplitude = AMPLITUDE_PERCENTILE of |x - median|
    """
    data = np.asarray(data, dtype=np.float64)
    dev = np.abs(data - np.median(data, axis=0, keepdims=True))
    noise = np.median(dev, axis=0) / 0.6745
    amplitude = np.percentile(dev, AMPLITUDE_PERCENTILE, axis=0)
    return noise, amplitude


def _robust_z(values):
    """Robust z of log(values); non-positive values (flat/dead) get z = -inf."""
    values = np.asarray(values, dtype=np.float64)
    z = np.full(values.shape, -np.inf)
    ok = values > 0
    if ok.sum() < 3:
        return z
    logv = np.log(values[ok])
    med = np.median(logv)
    scale = 1.4826 * np.median(np.abs(logv - med))
    if scale <= 0:
        scale = np.std(logv)
    z[ok] = (logv - med) / scale if scale > 0 else 0.0
    return z


def flag_bad_channels(noise, amplitude, z_threshold=Z_THRESHOLD):
    """Boolean mask: noise or amplitude is a robust outlier (either direction).

    Low outliers are dead/disconnected channels, high outliers are noisy or
    artifact-dominated ones.
    """
    return (np.abs(_robust_z(noise)) > z_threshold) | (np.abs(_robust_z(amplitude)) > z_threshold)
