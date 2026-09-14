"""Tests for network burst propagation (Epic D).

The core check is a synthetic travelling wave with a known speed: the fitted
speed has to come back, and a randomly ordered control has to fail to produce
one.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from propagation import burst_latencies, compute_propagation

PITCH_UM = 50.0
SPEED_UM_PER_MS = 100.0


def _travelling_wave(n_units=20, n_bursts=10, jitter_s=0.0002, seed=0):
    """Units on a line, each joining the burst after a fixed delay."""
    rng = np.random.default_rng(seed)
    locations = {f"u{i}": (i * PITCH_UM, 0.0) for i in range(n_units)}
    spikes = {f"u{i}": [] for i in range(n_units)}
    events = []

    for centre in np.arange(5.0, 5.0 + 10.0 * n_bursts, 10.0):
        for index in range(n_units):
            delay_s = (index * PITCH_UM / SPEED_UM_PER_MS) / 1000.0
            spikes[f"u{index}"].append(centre + delay_s + rng.normal(0, jitter_s))
        events.append({"start_time_s": centre - 0.01, "end_time_s": centre + 0.05})

    return {k: np.sort(np.asarray(v)) for k, v in spikes.items()}, events, locations


def _random_order(n_units=20, n_bursts=10, seed=1):
    rng = np.random.default_rng(seed)
    locations = {f"u{i}": (i * PITCH_UM, 0.0) for i in range(n_units)}
    spikes = {f"u{i}": [] for i in range(n_units)}
    events = []
    for centre in np.arange(5.0, 5.0 + 10.0 * n_bursts, 10.0):
        for index in range(n_units):
            spikes[f"u{index}"].append(centre + rng.uniform(0, 0.01))
        events.append({"start_time_s": centre - 0.01, "end_time_s": centre + 0.05})
    return {k: np.sort(np.asarray(v)) for k, v in spikes.items()}, events, locations


# ---------------------------------------------------------------------------
# Latencies
# ---------------------------------------------------------------------------

def test_latencies_are_measured_from_the_first_unit_to_fire():
    """Not from the detector's burst start, so the numbers stay comparable
    between detectors and thresholds."""
    spikes, events, _ = _travelling_wave(n_units=5, n_bursts=1, jitter_s=0.0)
    per_burst = burst_latencies(spikes, events)

    assert len(per_burst) == 1
    assert min(per_burst[0].values()) == pytest.approx(0.0)
    assert per_burst[0]["u4"] == pytest.approx(4 * PITCH_UM / SPEED_UM_PER_MS / 1000.0)


def test_bursts_with_too_few_participants_are_skipped():
    spikes, events, _ = _travelling_wave(n_units=3, n_bursts=4, jitter_s=0.0)
    assert burst_latencies(spikes, events, min_units_per_burst=5) == []
    assert len(burst_latencies(spikes, events, min_units_per_burst=3)) == 4


def test_only_the_first_spike_inside_the_window_counts():
    spikes = {"u0": np.array([1.0, 1.001, 1.002]), "u1": np.array([1.01])}
    events = [{"start_time_s": 0.99, "end_time_s": 1.05}]
    per_burst = burst_latencies(spikes, events, min_units_per_burst=2)
    assert per_burst[0]["u1"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def test_the_known_wave_speed_is_recovered():
    spikes, events, locations = _travelling_wave()
    summary = compute_propagation(spikes, events, unit_locations=locations)["summary"]

    assert summary["n_bursts_analysed"] == 10
    assert summary["median_speed_um_per_ms"] == pytest.approx(SPEED_UM_PER_MS, rel=0.05)
    assert summary["median_fit_r_squared"] > 0.9
    assert summary["fraction_bursts_with_speed_fit"] == pytest.approx(1.0)


def test_randomly_ordered_firing_yields_few_speed_fits():
    spikes, events, locations = _random_order()
    summary = compute_propagation(spikes, events, unit_locations=locations)["summary"]
    assert summary["fraction_bursts_with_speed_fit"] < 0.5


def test_speeds_are_skipped_without_locations_but_latencies_are_not():
    spikes, events, _ = _travelling_wave()
    result = compute_propagation(spikes, events, unit_locations=None)

    assert "median_speed_um_per_ms" not in result["summary"]
    assert result["units"]["u0"]["prop_leader_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Leader scores
# ---------------------------------------------------------------------------

def test_leader_scores_order_the_units_along_the_wave():
    spikes, events, locations = _travelling_wave()
    units = compute_propagation(spikes, events, unit_locations=locations)["units"]

    assert units["u0"]["prop_leader_score"] == pytest.approx(1.0)
    assert units["u19"]["prop_leader_score"] == pytest.approx(0.0)
    assert 0.4 < units["u10"]["prop_leader_score"] < 0.6


def test_leader_scores_spread_more_for_a_wave_than_for_random_order():
    wave_spikes, wave_events, locations = _travelling_wave()
    random_spikes, random_events, _ = _random_order()

    wave = compute_propagation(wave_spikes, wave_events, unit_locations=locations)
    shuffled = compute_propagation(random_spikes, random_events, unit_locations=locations)

    assert wave["summary"]["leader_score_std"] > 2 * shuffled["summary"]["leader_score_std"]


def test_per_unit_latency_reflects_position_in_the_wave():
    spikes, events, locations = _travelling_wave()
    units = compute_propagation(spikes, events, unit_locations=locations)["units"]
    expected_ms = 19 * PITCH_UM / SPEED_UM_PER_MS
    assert units["u19"]["prop_mean_latency_ms"] == pytest.approx(expected_ms, rel=0.05)
    assert units["u19"]["prop_participation_fraction"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_no_events_is_not_an_error():
    spikes, _, locations = _travelling_wave()
    result = compute_propagation(spikes, [], unit_locations=locations)
    assert result["summary"]["n_bursts_analysed"] == 0


def test_bursts_nobody_joins_are_reported_as_such():
    spikes, _, locations = _travelling_wave()
    far_away = [{"start_time_s": 500.0, "end_time_s": 501.0}]
    summary = compute_propagation(spikes, far_away, unit_locations=locations)["summary"]
    assert summary["n_bursts_analysed"] == 0
    assert summary["n_bursts_total"] == 1
