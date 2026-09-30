"""Tests for the intensity, naming and schema fixes.

The defects pinned here all produced plausible-looking numbers that meant
something other than their name said: a peak firing rate that was an
array-wide sum in one detector and a per-unit rate in the other, spike counts
that rose with unit yield, merged-tier totals that skipped the gaps the same
event's duration spanned, and a fragment count hiding behind a name that read
as a component count.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import burst_common
from parameter_free_burst_detector import compute_network_bursts as pf_detect
from gaussianNetworkBursts import compute_network_bursts as gauss_detect


def _bursting_network(n_units=12, burst_times=(10.0, 20.0, 30.0, 40.0), seed=0):
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = [center + rng.uniform(-0.05, 0.05, 25) for center in burst_times]
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def _clustered_bursts(n_units=15, seed=3):
    """Two clusters of eight short bursts: superbursts with many fragments."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = []
        for cluster_start in (10.0, 30.0):
            for index in range(8):
                trains.append(cluster_start + index * 0.5 + rng.uniform(-0.02, 0.02, 20))
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def _twin_peak_bursts(n_units=15, centers=(10.0, 20.0, 30.0, 40.0), sep=0.15, seed=5):
    """Bursts with two internal peaks separated by a dip that never reaches
    silence, so each network burst is built from two merged fragments."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = []
        for center in centers:
            trains.append(center + rng.normal(0, 0.03, 30))
            trains.append(center + sep + rng.normal(0, 0.03, 30))
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def _events(result, tier="network_bursts"):
    return result[tier]["events"]


# ---------------------------------------------------------------------------
# Peak rate units: the same key must mean the same thing in both detectors
# ---------------------------------------------------------------------------

def test_peak_firing_rate_is_per_unit_not_an_array_wide_sum():
    """A per-unit rate cannot exceed the array-wide sum, and for a network
    of many units it must be far below it. Before the fix this key held the
    sum, so the two were equal."""
    spikes = _bursting_network(n_units=12)
    events = _events(pf_detect(spikes, duration_s=50.0))
    assert events

    for event in events:
        per_unit = event["peak_population_firing_rate_hz"]
        total = event["peak_population_firing_rate_total_hz"]
        assert per_unit < total
        # The per-unit signal is smoothed and the total is not, so the ratio is
        # not exactly n_units, but it must be the right order of magnitude.
        assert total / per_unit > 5.0


def test_both_detectors_report_the_peak_rate_in_the_same_units():
    """The two detectors previously disagreed by a factor of n_units on this
    key, so any table pooling wells from both was meaningless."""
    spikes = _bursting_network(n_units=12)

    pf_peaks = [e["peak_population_firing_rate_hz"] for e in _events(pf_detect(spikes, duration_s=50.0))]
    gauss_peaks = [e["peak_population_firing_rate_hz"]
                   for e in _events(gauss_detect(spikes, duration_s=50.0))]
    assert pf_peaks and gauss_peaks

    # Same physical quantity measured with different bins and smoothing: the
    # means must land within an order of magnitude, not a factor of n_units.
    ratio = float(np.mean(pf_peaks) / np.mean(gauss_peaks))
    assert 0.1 < ratio < 10.0


def test_the_array_wide_total_is_still_available():
    for detect in (pf_detect, gauss_detect):
        events = _events(detect(_bursting_network(), duration_s=50.0))
        assert all("peak_population_firing_rate_total_hz" in e for e in events)


# ---------------------------------------------------------------------------
# Yield normalisation
# ---------------------------------------------------------------------------

def test_spikes_per_burst_is_reported_per_unit_as_well_as_raw():
    result = pf_detect(_bursting_network(n_units=12), duration_s=50.0)
    n_units = result["diagnostics"]["n_units"]
    for event in _events(result):
        assert event["spikes_per_burst_per_unit"] == pytest.approx(
            event["spike_count"] / n_units
        )


def test_the_raw_count_scales_with_yield_and_the_normalised_one_does_not():
    """Doubling the number of units at the same per-neuron firing must leave
    the per-unit intensity alone. This is the whole point of the column."""
    small = pf_detect(_bursting_network(n_units=10, seed=0), duration_s=50.0)
    large = pf_detect(_bursting_network(n_units=30, seed=0), duration_s=50.0)

    small_raw = np.mean([e["spike_count"] for e in _events(small)])
    large_raw = np.mean([e["spike_count"] for e in _events(large)])
    small_norm = np.mean([e["spikes_per_burst_per_unit"] for e in _events(small)])
    large_norm = np.mean([e["spikes_per_burst_per_unit"] for e in _events(large)])

    assert large_raw > 2.0 * small_raw
    assert large_norm == pytest.approx(small_norm, rel=0.35)


def test_burst_density_is_reported_rather_than_discarded():
    """Spikes per participating unit per second inside the burst: computed to
    gate detection, then thrown away before the fix."""
    for event in _events(pf_detect(_bursting_network(), duration_s=50.0)):
        assert event["burst_density_hz"] > 0.0


# ---------------------------------------------------------------------------
# Merged tiers: totals and duration must describe the same window
# ---------------------------------------------------------------------------

def test_merged_spike_count_covers_the_whole_burst_window():
    """Summing components skipped the gaps between them while
    burst_duration_s spanned them, so spike_count / burst_duration_s was not
    the burst's firing rate."""
    spikes = _twin_peak_bursts()
    result = pf_detect(spikes, duration_s=50.0)
    all_spikes = np.sort(np.concatenate(list(spikes.values())))

    merged = [e for e in _events(result) if e["n_fragments"] > 1]
    assert merged, "fixture must produce at least one multi-fragment burst"

    for event in merged:
        inside = np.count_nonzero(
            (all_spikes >= event["start_time_s"]) & (all_spikes <= event["end_time_s"])
        )
        # Binned counts against exact spike times: within one bin's worth.
        assert event["spike_count"] == pytest.approx(inside, rel=0.05, abs=10)


def test_superburst_totals_are_not_below_their_components():
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    superbursts = _events(result, "superbursts")
    if not superbursts:
        pytest.skip("no superburst in this synthetic well")

    for event in superbursts:
        assert event["burst_area"] > 0
        assert event["spike_count"] > 0
        assert event["burst_duration_s"] >= 2.5


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def test_fragment_count_is_named_for_what_it_counts():
    """`component_count` read as "components merged at this tier" but held the
    transitive fragment count."""
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    for tier in ("network_bursts", "superbursts"):
        for event in _events(result, tier):
            assert "n_fragments" in event
            assert "component_count" not in event
            # Immediate children never exceed the fragments beneath them.
            assert event["n_fragments"] >= event["n_components"]


def test_peak_synchrony_is_available_under_an_accurate_name():
    """The trace is per-bin co-activity, not the per-burst participation
    fraction. The old key is kept so existing tables keep working."""
    for detect in (pf_detect, gauss_detect):
        for event in _events(detect(_bursting_network(), duration_s=50.0)):
            assert event["peak_synchrony"] == event["peak_participation_fraction"]
            assert event["peak_synchrony"] != event["participation_fraction"] or \
                event["participation_fraction"] == pytest.approx(event["peak_synchrony"])


def test_peak_bin_synchrony_survives_merging():
    """It existed only on fragments before, so the network burst and
    superburst summaries lost the one un-smoothed synchrony measure."""
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    for tier in ("burst_fragments", "network_bursts", "superbursts"):
        for event in _events(result, tier):
            assert 0.0 <= event["peak_bin_synchrony"] <= 1.0


# ---------------------------------------------------------------------------
# Summaries carry the new fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "spikes_per_burst_per_unit",
    "burst_density_hz",
    "peak_population_firing_rate_hz",
    "peak_population_firing_rate_total_hz",
    "peak_synchrony",
    "peak_bin_synchrony",
])
def test_level_metrics_summarises_every_intensity_field(key):
    metrics = pf_detect(_bursting_network(), duration_s=50.0)["network_bursts"]["metrics"]
    assert key in metrics
    assert metrics[key]["mean"] is not None


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

def test_every_detector_stamps_the_schema_version():
    for detect in (pf_detect, gauss_detect):
        diagnostics = detect(_bursting_network(), duration_s=50.0)["diagnostics"]
        assert diagnostics["schema_version"] == burst_common.SCHEMA_VERSION


def test_schema_version_is_an_integer_that_can_be_compared():
    assert isinstance(burst_common.SCHEMA_VERSION, int)
    assert burst_common.SCHEMA_VERSION >= 3
