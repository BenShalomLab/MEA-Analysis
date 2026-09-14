"""Regression tests for the Epic A correctness fixes.

Each test pins one behaviour that was wrong before and would silently revert:
undefined metrics reported as zero, rates divided by the spike span instead of
the recording duration, single bursts labelled superbursts, a Gaussian
detector with no height gate and an inverted edge rule, a curation rule that
depended on SpikeInterface's amplitude sign convention, and a local
common-reference annulus that selected no channels at all.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import burst_common
from parameter_free_burst_detector import compute_network_bursts as pf_detect
from gaussianNetworkBursts import compute_network_bursts as gauss_detect


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _bursting_network(n_units=12, burst_times=(10.0, 20.0, 30.0, 40.0), seed=0):
    """Tight synchronous bursts on a low-rate background."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = [center + rng.uniform(-0.05, 0.05, 25) for center in burst_times]
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def _asynchronous_network(n_units=12, duration=50.0, rate_hz=1.0, seed=1):
    """Independent Poisson units: no network bursts exist in this data."""
    rng = np.random.default_rng(seed)
    return {
        f"u{u}": np.sort(rng.uniform(0, duration, int(rate_hz * duration)))
        for u in range(n_units)
    }


def _clusters_and_one_long_burst(n_units=15, seed=3):
    """Two clusters of eight short bursts, plus one sustained 4 s epoch.

    The clusters are superbursts under either definition. The sustained epoch
    is a single long network burst: a superburst only if one component is
    allowed to count, which is the behaviour this fix changed.
    """
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = []
        for cluster_start in (10.0, 30.0):
            for k in range(8):
                trains.append(cluster_start + k * 0.5 + rng.uniform(-0.02, 0.02, 20))
        trains.append(rng.uniform(55.0, 59.0, 400))
        trains.append(rng.uniform(0, 70, 4))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


# ---------------------------------------------------------------------------
# burst_common: undefined metrics must not be reported as zero
# ---------------------------------------------------------------------------

def test_stats_of_empty_input_is_null_not_zero():
    empty = burst_common.stats([])
    assert empty == {"mean": None, "std": None, "cv": None}


def test_stats_cv_is_null_when_mean_is_zero():
    assert burst_common.stats([-1.0, 1.0])["cv"] is None


def test_level_metrics_with_no_events_reports_a_zero_count_not_zero_metrics():
    metrics = burst_common.level_metrics([], total_dur=100.0)
    assert metrics["burst_count"] == 0
    assert metrics["burst_rate_hz"] == 0.0
    # Crucially there is no burst_duration_s of 0.0 to average into a group.
    assert "burst_duration_s" not in metrics


def test_level_metrics_reports_both_ibi_conventions_and_the_duration_tail():
    events = [
        {"start_time_s": 0.0, "end_time_s": 1.0, "burst_duration_s": 1.0},
        {"start_time_s": 10.0, "end_time_s": 11.0, "burst_duration_s": 1.0},
        {"start_time_s": 20.0, "end_time_s": 26.0, "burst_duration_s": 6.0},
    ]
    metrics = burst_common.level_metrics(events, total_dur=100.0)

    # Onset-to-onset is the burst cycle period: 10 s and 10 s.
    assert metrics["ibi_s"]["mean"] == pytest.approx(10.0)
    # Offset-to-onset is the silent gap: 9 s and 9 s.
    assert metrics["ibi_gap_s"]["mean"] == pytest.approx(9.0)
    # The long burst shows up in the tail even though the mean is pulled down.
    assert metrics["burst_duration_max_s"] == pytest.approx(6.0)
    assert metrics["burst_duration_p95_s"] > metrics["burst_duration_s"]["mean"]


def test_level_metrics_rate_uses_the_duration_it_is_given():
    events = [{"start_time_s": 0.0, "end_time_s": 1.0, "burst_duration_s": 1.0}]
    active_span = burst_common.level_metrics(events, total_dur=10.0)["burst_rate_hz"]
    whole_record = burst_common.level_metrics(events, total_dur=100.0)["burst_rate_hz"]
    assert active_span == pytest.approx(10 * whole_record)


# ---------------------------------------------------------------------------
# A3: rates divide by recording duration, not the first-to-last-spike span
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detect", [pf_detect, gauss_detect])
def test_rates_fall_when_the_recording_is_longer_than_the_active_span(detect):
    spikes = _bursting_network()
    span = max(t.max() for t in spikes.values()) - min(t.min() for t in spikes.values())
    recording_duration = 4 * span

    without = detect(SpikeTimes=spikes)
    with_duration = detect(SpikeTimes=spikes, duration_s=recording_duration)

    assert without["diagnostics"]["duration_source"] == "spike_span"
    assert with_duration["diagnostics"]["duration_source"] == "recording"
    assert with_duration["diagnostics"]["recording_duration_s"] == pytest.approx(recording_duration)

    # Same bursts detected either way; only the denominator changed.
    assert (len(with_duration["network_bursts"]["events"])
            == len(without["network_bursts"]["events"]))

    rate_with = with_duration["network_bursts"]["metrics"]["burst_rate_hz"]
    rate_without = without["network_bursts"]["metrics"]["burst_rate_hz"]
    assert rate_with < rate_without
    assert rate_with == pytest.approx(rate_without * span / recording_duration, rel=1e-6)


def test_unit_firing_rate_uses_the_recording_duration():
    spikes = {"u0": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    result = pf_detect(SpikeTimes=spikes, duration_s=100.0)
    # 5 spikes over 100 s, not over the 4 s between the first and last spike.
    assert result["unit_stats"]["u0"]["mean_firing_rate_hz"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# A4: a superburst is a cluster of network bursts, not one long burst
# ---------------------------------------------------------------------------

def test_a_single_long_burst_is_not_a_superburst():
    spikes = _clusters_and_one_long_burst()
    result = pf_detect(SpikeTimes=spikes, duration_s=70.0)

    assert result["diagnostics"]["superburst_min_components"] == 2
    events = result["superbursts"]["events"]
    assert len(events) == 2, "expected only the two genuine burst clusters"
    assert all(event["n_components"] >= 2 for event in events)

    # The long burst is still reported, as a long network burst.
    assert result["network_bursts"]["metrics"]["burst_duration_max_s"] > 2.5


def test_single_component_superbursts_can_be_re_enabled():
    """The old behaviour stays reachable, it is just no longer the default."""
    spikes = _clusters_and_one_long_burst()
    strict = pf_detect(SpikeTimes=spikes, duration_s=70.0)
    permissive = pf_detect(SpikeTimes=spikes, duration_s=70.0, min_superburst_components=1)

    assert len(permissive["superbursts"]["events"]) == len(strict["superbursts"]["events"]) + 1
    assert any(e["n_components"] == 1 for e in permissive["superbursts"]["events"])


# ---------------------------------------------------------------------------
# A5: Gaussian detector height gate and edge rule
# ---------------------------------------------------------------------------

def test_height_gate_rejects_an_asynchronous_network():
    spikes = _asynchronous_network()

    gated = gauss_detect(SpikeTimes=spikes, duration_s=50.0)
    ungated = gauss_detect(SpikeTimes=spikes, duration_s=50.0, min_height_sd=None)

    assert gated["diagnostics"]["detection_threshold_hz"] is not None
    assert ungated["diagnostics"]["detection_threshold_hz"] is None
    # Poisson firing has no network bursts; without a height gate the
    # prominence floor scales with the well's own noise and finds them anyway.
    assert len(gated["network_bursts"]["events"]) < len(ungated["network_bursts"]["events"])


def test_height_gate_still_finds_real_bursts():
    result = gauss_detect(SpikeTimes=_bursting_network(), duration_s=50.0)
    assert len(result["network_bursts"]["events"]) >= 3


def test_zero_or_negative_height_sd_disables_the_gate():
    spikes = _bursting_network()
    for disabled in (0, -1):
        result = gauss_detect(SpikeTimes=spikes, duration_s=50.0, min_height_sd=disabled)
        assert result["diagnostics"]["detection_threshold_hz"] is None
        assert result["diagnostics"]["min_height_sd"] is None


def test_burst_edges_sit_at_a_fraction_of_the_peak_not_at_one_minus_it():
    """Edges are where the rate falls to frac*peak, so a smaller fraction
    gives a longer burst. The previous rule used (1 - frac)*peak, which
    clipped every burst to its crest and shortened durations several-fold."""
    spikes = _bursting_network()

    low_edge = gauss_detect(SpikeTimes=spikes, duration_s=50.0, onset_offset_peak_frac=0.1)
    high_edge = gauss_detect(SpikeTimes=spikes, duration_s=50.0, onset_offset_peak_frac=0.7)

    assert low_edge["diagnostics"]["edge_rule"] == "fraction_of_peak"
    assert (low_edge["network_bursts"]["metrics"]["burst_duration_s"]["mean"]
            > high_edge["network_bursts"]["metrics"]["burst_duration_s"]["mean"])


def test_detected_burst_spans_the_width_at_the_edge_level():
    """Direct check against the smoothed trace the detector returns."""
    frac = 0.3
    result = gauss_detect(SpikeTimes=_bursting_network(), duration_s=50.0,
                          onset_offset_peak_frac=frac)
    assert result["network_bursts"]["events"], "no bursts to check"

    time_s = np.asarray(result["plot_data"]["time_s"])
    rate = np.asarray(result["plot_data"]["population_firing_rate_hz"])
    event = result["network_bursts"]["events"][0]

    peak_value = event["peak_population_firing_rate_hz"]
    inside = (time_s >= event["start_time_s"]) & (time_s <= event["end_time_s"])
    # Everything strictly inside the burst is above the edge level, i.e. the
    # burst extends down to 30% of its peak rather than stopping at 70%.
    assert rate[inside].min() <= peak_value * frac * 1.05
    assert rate[inside].max() == pytest.approx(peak_value)


# ---------------------------------------------------------------------------
# A6: curation rules and the rejection audit
# ---------------------------------------------------------------------------

def _curation_harness():
    """A bare object carrying just the curation method and a logger."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("spikeinterface")
    from mea_reports import ReportsMixin

    harness = types.SimpleNamespace()
    harness.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    harness._apply_curation_logic = types.MethodType(
        ReportsMixin._apply_curation_logic, harness
    )
    return harness, pd


def test_amplitude_rule_works_for_both_sign_conventions():
    harness, pd = _curation_harness()
    thresholds = {"amplitude_median": -20}

    # Same three units, reported signed (older SpikeInterface) and as a
    # magnitude (newer). Curation must keep and reject the same units.
    base = {"presence_ratio": [1.0, 1.0, 1.0],
            "rp_contamination": [0.0, 0.0, 0.0],
            "firing_rate": [1.0, 1.0, 1.0]}
    signed = pd.DataFrame({**base, "amplitude_median": [-50.0, -30.0, -5.0]},
                          index=["big", "ok", "small"])
    magnitude = pd.DataFrame({**base, "amplitude_median": [50.0, 30.0, 5.0]},
                             index=["big", "ok", "small"])

    kept_signed, _ = harness._apply_curation_logic(signed, thresholds)
    kept_magnitude, _ = harness._apply_curation_logic(magnitude, thresholds)

    assert list(kept_signed.index) == ["big", "ok"]
    assert list(kept_magnitude.index) == ["big", "ok"]


def test_amplitude_cv_rule_is_actually_applied():
    harness, pd = _curation_harness()
    metrics = pd.DataFrame(
        {
            "presence_ratio": [1.0, 1.0],
            "rp_contamination": [0.0, 0.0],
            "firing_rate": [1.0, 1.0],
            "amplitude_median": [-50.0, -50.0],
            "amplitude_cv_median": [0.2, 0.9],
        },
        index=["stable", "unstable"],
    )
    kept, _ = harness._apply_curation_logic(metrics, {"amplitude_cv_median": 0.5})
    assert list(kept.index) == ["stable"]
    assert harness.curation_summary["rejected_by_reason"]["Unstable Amp"] == 1


def test_curation_summary_counts_every_rejection_reason():
    harness, pd = _curation_harness()
    metrics = pd.DataFrame(
        {
            "presence_ratio": [1.0, 0.1, 1.0, 1.0],
            "rp_contamination": [0.0, 0.0, 0.9, 0.0],
            "firing_rate": [1.0, 1.0, 1.0, 0.0001],
            "amplitude_median": [-50.0, -50.0, -50.0, -50.0],
        },
        index=["good", "absent", "contaminated", "silent"],
    )
    kept, rejections = harness._apply_curation_logic(metrics, None)

    assert list(kept.index) == ["good"]
    summary = harness.curation_summary
    assert summary["n_units_input"] == 4
    assert summary["n_units_kept"] == 1
    assert summary["n_units_rejected"] == 3
    assert summary["rejected_by_reason"] == {
        "Low Presence": 1, "High Contam": 1, "Low FR": 1,
    }
    assert len(rejections) == 3


def test_curation_records_rules_it_could_not_apply():
    harness, pd = _curation_harness()
    metrics = pd.DataFrame({"firing_rate": [1.0]}, index=["u0"])
    harness._apply_curation_logic(metrics, None)
    skipped = harness.curation_summary["rules_skipped_missing_metric"]
    assert "presence_ratio" in skipped and "amplitude_median" in skipped


# ---------------------------------------------------------------------------
# A2: the local common-reference annulus must actually select channels
# ---------------------------------------------------------------------------

def _coverage_harness(locations):
    pytest.importorskip("spikeinterface")
    from mea_preprocessing import PreprocessingMixin

    harness = types.SimpleNamespace()
    harness.logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    harness._local_reference_coverage = types.MethodType(
        PreprocessingMixin._local_reference_coverage, harness
    )
    recording = types.SimpleNamespace(
        get_channel_locations=lambda: np.asarray(locations, dtype=float)
    )
    return harness, recording


def _grid_locations(n=8, pitch=60.0):
    return [(x * pitch, y * pitch) for x in range(n) for y in range(n)]


def test_equal_inner_and_outer_radius_selects_no_neighbours():
    """The shipped configuration was local_radius=(250, 250): an empty
    annulus, so every channel was left unreferenced and the step did
    nothing."""
    harness, recording = _coverage_harness(_grid_locations())
    covered, median_neighbours = harness._local_reference_coverage(recording, 250.0, 250.0)
    assert covered == 0.0
    assert median_neighbours == 0.0


def test_configured_radii_cover_a_sparse_grid():
    from mea_preprocessing import CMR_INNER_RADIUS_UM, CMR_OUTER_RADIUS_UM

    harness, recording = _coverage_harness(_grid_locations())
    covered, median_neighbours = harness._local_reference_coverage(
        recording, CMR_INNER_RADIUS_UM, CMR_OUTER_RADIUS_UM
    )
    assert covered == 1.0
    assert median_neighbours >= 4


def test_isolated_channels_are_reported_as_uncovered():
    harness, recording = _coverage_harness([(0.0, 0.0), (10000.0, 10000.0)])
    covered, _ = harness._local_reference_coverage(recording, 30.0, 200.0)
    assert covered == 0.0


def test_coverage_is_none_without_channel_locations():
    harness, _ = _coverage_harness([])
    recording = types.SimpleNamespace(
        get_channel_locations=lambda: (_ for _ in ()).throw(ValueError("no probe"))
    )
    assert harness._local_reference_coverage(recording, 30.0, 200.0) is None
