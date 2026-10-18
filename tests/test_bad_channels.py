import numpy as np

from bad_channels import channel_noise_amplitude, flag_bad_channels


def test_flags_noisy_dead_and_artifact_channels():
    rng = np.random.default_rng(0)
    n_ch = 60
    data = rng.normal(0, 5.0, size=(20000, n_ch))
    data[:, 3] *= 8          # noisy
    data[:, 7] = 0.0         # dead / flat
    data[::500, 11] += 400   # amplitude artifact, noise unchanged
    noise, amp = channel_noise_amplitude(data)
    bad = set(np.flatnonzero(flag_bad_channels(noise, amp)))
    assert {3, 7, 11} <= bad
    assert len(bad) <= 5


def test_clean_array_flags_nothing():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 5.0, size=(20000, 60))
    noise, amp = channel_noise_amplitude(data)
    assert not flag_bad_channels(noise, amp).any()
