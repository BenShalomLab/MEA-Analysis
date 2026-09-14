"""Tests for network burst shape and spike participation (Epic C, C5/C6)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import burst_common
from gaussianNetworkBursts import compute_network_bursts as gauss_detect
from parameter_free_burst_detector import compute_network_bursts as pf_detect


def _bursting_network(n_units=12, burst_times=(10.0, 20.0, 30.0, 40.0), seed=0):
    rng = np.random.default_rng(seed)
    spikes = {}
    for unit in range(n_units):
        trains = [center + rng.uniform(-0.05, 0.05, 25) for center in burst_times]
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{unit}"] = np.sort(np.concatenate(trains))
    return spikes


# ---------------------------------------------------------------------------
# Spike participation
# ---------------------------------------------------------------------------

def test_participation_counts_spikes_inside_event_windows():
    spikes = np.array([0.0, 0.5, 1.0, 5.0, 10.2, 10.5, 20.0])
    events = [
        {"start_time_s": 0.0, "end_time_s": 1.0},
        {"start_time_s": 10.0, "end_time_s": 11.0},
    ]
    result = burst_common.spike_participation(spikes, events)

    assert result["n_spikes_total"] == 7
    assert result["n_spikes_in_network_bursts"] == 5
    assert result["fraction_spikes_in_network_bursts"] == pytest.approx(5 / 7)
    assert result["percent_random_spikes"] == pytest.approx(100 * 2 / 7)


def test_overlapping_windows_do_not_double_count():
    spikes = np.array([1.0, 1.5, 2.0])
    events = [
        {"start_time_s": 0.0, "end_time_s": 2.0},
        {"start_time_s": 1.0, "end_time_s": 3.0},
    ]
    result = burst_common.spike_participation(spikes, events)
    assert result["n_spikes_in_network_bursts"] == 3


def test_no_events_means_every_spike_is_random():
    result = burst_common.spike_participation(np.array([1.0, 2.0]), [])
    assert result["fraction_spikes_in_network_bursts"] == 0.0
    assert result["percent_random_spikes"] == 100.0


def test_no_spikes_gives_null_fractions_not_zero():
    result = burst_common.spike_participation(np.array([]), [])
    assert result["n_spikes_total"] == 0
    assert result["fraction_spikes_in_network_bursts"] is None


@pytest.mark.parametrize("detect", [pf_detect, gauss_detect])
def test_detectors_report_spike_participation(detect):
    result = detect(SpikeTimes=_bursting_network(), duration_s=50.0)
    participation = result["spike_participation"]

    assert participation["n_spikes_total"] == 12 * (4 * 25 + 5)
    # The synthetic units fire 100 of 105 spikes inside the four bursts.
    assert participation["fraction_spikes_in_network_bursts"] > 0.5
    assert participation["percent_random_spikes"] == pytest.approx(
        100 * (1 - participation["fraction_spikes_in_network_bursts"])
    )


# ---------------------------------------------------------------------------
# Burst shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detect", [pf_detect, gauss_detect])
def test_rise_and_decay_split_the_burst_at_its_peak(detect):
    result = detect(SpikeTimes=_bursting_network(), duration_s=50.0)
    events = result["network_bursts"]["events"]
    assert events

    for event in events:
        assert event["rise_time_s"] >= 0
        assert event["decay_time_s"] >= 0
        assert (event["rise_time_s"] + event["decay_time_s"]) == pytest.approx(
            event["burst_duration_s"], rel=1e-6
        )


@pytest.mark.parametrize("detect", [pf_detect, gauss_detect])
def test_shape_metrics_are_summarised_across_bursts(detect):
    metrics = detect(SpikeTimes=_bursting_network(), duration_s=50.0)["network_bursts"]["metrics"]
    assert metrics["rise_time_s"]["mean"] >= 0
    assert metrics["decay_time_s"]["mean"] >= 0


# ---------------------------------------------------------------------------
# Duty cycle
# ---------------------------------------------------------------------------

def test_duty_cycle_is_the_share_of_time_spent_bursting():
    # 1 s bursts with 9 s silences: the network bursts 10% of the time.
    events = [
        {"start_time_s": float(i * 10), "end_time_s": float(i * 10 + 1),
         "burst_duration_s": 1.0}
        for i in range(5)
    ]
    metrics = burst_common.level_metrics(events, total_dur=50.0)
    assert metrics["duty_cycle"] == pytest.approx(0.1)


def test_duty_cycle_rises_when_bursts_lengthen():
    short = [{"start_time_s": float(i * 10), "end_time_s": float(i * 10 + 1),
              "burst_duration_s": 1.0} for i in range(5)]
    long = [{"start_time_s": float(i * 10), "end_time_s": float(i * 10 + 5),
             "burst_duration_s": 5.0} for i in range(5)]

    assert (burst_common.level_metrics(long, 50.0)["duty_cycle"]
            > burst_common.level_metrics(short, 50.0)["duty_cycle"])


def test_duty_cycle_is_absent_for_a_single_burst():
    """One burst gives no inter-burst interval, so the ratio is undefined."""
    events = [{"start_time_s": 0.0, "end_time_s": 1.0, "burst_duration_s": 1.0}]
    assert "duty_cycle" not in burst_common.level_metrics(events, total_dur=50.0)
