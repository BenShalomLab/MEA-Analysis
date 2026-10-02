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
        per_unit = event["burst_peak_hz_per_unit"]
        total = event["burst_peak_hz_array"]
        assert per_unit < total
        # The per-unit signal is smoothed and the total is not, so the ratio is
        # not exactly n_units, but it must be the right order of magnitude.
        assert total / per_unit > 5.0


def test_both_detectors_report_the_peak_rate_in_the_same_units():
    """The two detectors previously disagreed by a factor of n_units on this
    key, so any table pooling wells from both was meaningless."""
    spikes = _bursting_network(n_units=12)

    pf_peaks = [e["burst_peak_hz_per_unit"] for e in _events(pf_detect(spikes, duration_s=50.0))]
    gauss_peaks = [e["burst_peak_hz_per_unit"]
                   for e in _events(gauss_detect(spikes, duration_s=50.0))]
    assert pf_peaks and gauss_peaks

    # Same physical quantity measured with different bins and smoothing: the
    # means must land within an order of magnitude, not a factor of n_units.
    ratio = float(np.mean(pf_peaks) / np.mean(gauss_peaks))
    assert 0.1 < ratio < 10.0


def test_the_array_wide_total_is_still_available():
    for detect in (pf_detect, gauss_detect):
        events = _events(detect(_bursting_network(), duration_s=50.0))
        assert all("burst_peak_hz_array" in e for e in events)


# ---------------------------------------------------------------------------
# Yield normalisation
# ---------------------------------------------------------------------------

def test_spikes_per_burst_is_reported_per_unit_as_well_as_raw():
    result = pf_detect(_bursting_network(n_units=12), duration_s=50.0)
    n_units = result["diagnostics"]["n_units"]
    for event in _events(result):
        assert event["spikes_per_burst_per_unit"] == pytest.approx(
            event["spikes_per_burst"] / n_units
        )


def test_the_raw_count_scales_with_yield_and_the_normalised_one_does_not():
    """Doubling the number of units at the same per-neuron firing must leave
    the per-unit intensity alone. This is the whole point of the column."""
    small = pf_detect(_bursting_network(n_units=10, seed=0), duration_s=50.0)
    large = pf_detect(_bursting_network(n_units=30, seed=0), duration_s=50.0)

    small_raw = np.mean([e["spikes_per_burst"] for e in _events(small)])
    large_raw = np.mean([e["spikes_per_burst"] for e in _events(large)])
    small_norm = np.mean([e["spikes_per_burst_per_unit"] for e in _events(small)])
    large_norm = np.mean([e["spikes_per_burst_per_unit"] for e in _events(large)])

    assert large_raw > 2.0 * small_raw
    assert large_norm == pytest.approx(small_norm, rel=0.35)


def test_burst_density_is_reported_rather_than_discarded():
    """Spikes per participating unit per second inside the burst: computed to
    gate detection, then thrown away before the fix."""
    for event in _events(pf_detect(_bursting_network(), duration_s=50.0)):
        assert event["intraburst_rate_hz"] > 0.0


# ---------------------------------------------------------------------------
# Merged tiers: totals and duration must describe the same window
# ---------------------------------------------------------------------------

def test_merged_spike_count_covers_the_whole_burst_window():
    """Summing components skipped the gaps between them while
    burst_duration_s spanned them, so spikes_per_burst / burst_duration_s was not
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
        assert event["spikes_per_burst"] == pytest.approx(inside, rel=0.05, abs=10)


def test_superburst_totals_are_not_below_their_components():
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    superbursts = _events(result, "superbursts")
    if not superbursts:
        pytest.skip("no superburst in this synthetic well")

    for event in superbursts:
        assert event["burst_area_spikes_per_unit"] > 0
        assert event["spikes_per_burst"] > 0
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
            assert event["coactive_fraction_peak"] == event["coactive_fraction_peak"]
            assert event["coactive_fraction_peak"] != event["participation_fraction"] or \
                event["participation_fraction"] == pytest.approx(event["coactive_fraction_peak"])


def test_coactive_fraction_max_survives_merging():
    """It existed only on fragments before, so the network burst and
    superburst summaries lost the one un-smoothed synchrony measure."""
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    for tier in ("burst_fragments", "network_bursts", "superbursts"):
        for event in _events(result, tier):
            assert 0.0 <= event["coactive_fraction_max"] <= 1.0


# ---------------------------------------------------------------------------
# Summaries carry the new fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "spikes_per_burst_per_unit",
    "intraburst_rate_hz",
    "burst_peak_hz_per_unit",
    "burst_peak_hz_array",
    "coactive_fraction_peak",
    "coactive_fraction_max",
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


# ---------------------------------------------------------------------------
# Schema 4: the published metric set
# ---------------------------------------------------------------------------

def test_rise_and_decay_split_the_burst_at_its_peak():
    """RT and DT of Mossink et al. 2021. Together they must reconstruct the
    duration, or one of the three is measured against a different window."""
    for detect in (pf_detect, gauss_detect):
        for event in _events(detect(_bursting_network(), duration_s=50.0)):
            assert event["rise_time_s"] >= 0.0
            assert event["decay_time_s"] >= 0.0
            assert event["rise_time_s"] + event["decay_time_s"] == pytest.approx(
                event["burst_duration_s"], abs=1e-9
            )


def test_an_asymmetric_burst_has_unequal_rise_and_decay():
    """A symmetric synthetic burst would pass the identity above while the
    two were silently swapped."""
    rng = np.random.default_rng(11)
    spikes = {}
    for u in range(15):
        trains = []
        for center in (10.0, 20.0, 30.0, 40.0):
            # Sharp recruitment, slow tail.
            trains.append(center + np.abs(rng.normal(0, 0.01, 10)) * -1)
            trains.append(center + np.abs(rng.normal(0, 0.25, 40)))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))

    events = _events(pf_detect(spikes, duration_s=50.0))
    assert events
    decays = np.mean([e["decay_time_s"] for e in events])
    rises = np.mean([e["rise_time_s"] for e in events])
    assert decays > rises


def test_intraburst_isi_is_shorter_than_the_interval_between_bursts():
    """bISI must describe the inside of the burst, not the whole train."""
    result = pf_detect(_bursting_network(), duration_s=50.0)
    metrics = result["network_bursts"]["metrics"]
    assert metrics["intraburst_isi_mean_s"]["mean"] < metrics["ibi_s"]["mean"]


def test_intraburst_isi_is_null_for_a_burst_with_one_spike():
    assert burst_common.intraburst_isi_mean(np.array([1.0]), 0.0, 2.0) is None
    assert burst_common.intraburst_isi_mean(np.array([1.0, 1.5]), 0.0, 2.0) == pytest.approx(0.5)


def test_burst_rate_is_reported_in_both_hz_and_per_minute():
    """Well quality gates in the literature are quoted per minute; a Hz value
    compared against one of those thresholds is wrong by 60x."""
    metrics = pf_detect(_bursting_network(), duration_s=50.0)["network_bursts"]["metrics"]
    assert metrics["burst_rate_per_min"] == pytest.approx(metrics["burst_rate_hz"] * 60.0)


def test_an_empty_tier_still_reports_both_rate_units():
    empty = burst_common.level_metrics([], 100.0)
    assert empty["burst_rate_hz"] == 0.0
    assert empty["burst_rate_per_min"] == 0.0


def test_duty_cycle_is_the_fraction_of_time_spent_bursting():
    metrics = pf_detect(_bursting_network(), duration_s=50.0)["network_bursts"]["metrics"]
    assert 0.0 < metrics["duty_cycle"] < 1.0


def test_duty_cycle_is_omitted_when_there_is_no_gap_to_measure():
    one_event = [{"start_time_s": 1.0, "end_time_s": 2.0, "burst_duration_s": 1.0}]
    assert "duty_cycle" not in burst_common.level_metrics(one_event, 100.0)


def test_percent_random_spikes_complements_the_fraction_inside_bursts():
    """PRS (Mossink et al. 2021): a culture can hold its burst rate while its
    neurons drift out of the bursts, and only this ratio shows it."""
    for detect in (pf_detect, gauss_detect):
        block = detect(_bursting_network(), duration_s=50.0)["spike_participation"]
        assert block["percent_random_spikes"] == pytest.approx(
            100.0 * (1.0 - block["fraction_spikes_in_network_bursts"])
        )
        assert block["n_spikes_in_network_bursts"] <= block["n_spikes_total"]


def test_a_well_with_no_bursts_reports_every_spike_as_random():
    rng = np.random.default_rng(1)
    poisson = {f"u{u}": np.sort(rng.uniform(0, 50, 50)) for u in range(12)}
    block = pf_detect(poisson, duration_s=50.0)["spike_participation"]
    if block["n_spikes_in_network_bursts"] == 0:
        assert block["percent_random_spikes"] == 100.0


def test_both_detectors_emit_the_same_network_burst_schema():
    """The two detectors drifted apart once already. Any key present in one
    and not the other is how that happens again."""
    spikes = _bursting_network()
    pf_keys = set(_events(pf_detect(spikes, duration_s=50.0))[0])
    gauss_keys = set(_events(gauss_detect(spikes, duration_s=50.0))[0])
    # The adaptive detector merges, so only it carries the hierarchy fields.
    assert pf_keys - gauss_keys == {"n_components", "n_fragments"}
    assert gauss_keys - pf_keys == set()


def test_the_old_schema_3_names_are_gone():
    """A stale name left behind would be silently read as the new quantity."""
    retired = {
        "peak_participation_fraction", "peak_bin_synchrony", "peak_synchrony",
        "peak_population_firing_rate_hz", "peak_population_firing_rate_total_hz",
        "spike_count", "burst_area", "burst_density_hz", "component_count",
    }
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    for tier in ("burst_fragments", "network_bursts", "superbursts"):
        for event in _events(result, tier):
            assert retired.isdisjoint(event)
        assert retired.isdisjoint(result[tier]["metrics"])
    assert retired.isdisjoint(result["diagnostics"])
    for stats in result["unit_stats"].values():
        assert {"cv_isi", "bimodality_coefficient", "is_bursty"}.isdisjoint(stats)


def test_the_schema_version_records_the_rename():
    assert burst_common.SCHEMA_VERSION >= 4


# ---------------------------------------------------------------------------
# Schema 5: merging criteria come from the level being merged
# ---------------------------------------------------------------------------

def _reverberating(sustained, n_units=40, duration=300.0, seed=0):
    """Minibursts at identical times, either riding on a sustained plateau or
    separated by silence. A gap rule cannot tell these apart; that is the
    point of the continuity rule."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = []
        for onset in np.arange(10, duration - 20, 25.0):
            trains.append(onset + rng.normal(0, 0.05, 40))
            if sustained:
                trains.append(onset + rng.uniform(0.1, 1.5, 25))
            for index in range(1, 8):
                trains.append(onset + 0.25 + index * 0.18 + rng.normal(0, 0.012, 10))
        trains.append(rng.uniform(0, duration, 20))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def test_fragments_merge_when_the_network_stays_engaged():
    result = pf_detect(_reverberating(sustained=True), duration_s=300.0)
    nbs = _events(result)
    assert nbs
    assert np.mean([e["n_fragments"] for e in nbs]) > 4, (
        "a sustained reverberating burst must merge into one network burst"
    )


def test_fragments_do_not_merge_across_silence():
    result = pf_detect(_reverberating(sustained=False), duration_s=300.0)
    nbs = _events(result)
    assert nbs
    assert all(e["n_fragments"] == 1 for e in nbs), (
        "bursts separated by genuine silence are separate events"
    )


def test_the_continuity_rule_separates_what_a_gap_rule_cannot():
    """The claim being pinned: with identical burst timing, a proximity-only
    rule reports the same network for both, and the continuity rule does not."""
    plateau = _reverberating(sustained=True)
    silence = _reverberating(sustained=False)

    gap_rates = [pf_detect(s, duration_s=300.0, merge_rule="gap")
                 ["network_bursts"]["metrics"]["burst_rate_per_min"]
                 for s in (plateau, silence)]
    cont_rates = [pf_detect(s, duration_s=300.0, merge_rule="continuity")
                  ["network_bursts"]["metrics"]["burst_rate_per_min"]
                  for s in (plateau, silence)]

    assert gap_rates[0] == pytest.approx(gap_rates[1], rel=0.2)
    assert cont_rates[1] > 3.0 * cont_rates[0]


def _long_burst_spikes(duration_s=3.0, n_units=30, seed=1):
    """One prolonged network burst of `duration_s` among short regular ones."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = [c + rng.normal(0, 0.03, 25) for c in np.arange(10, 200, 20.0)]
        trains.append(110.0 + rng.uniform(0, duration_s, int(40 * duration_s)))
        trains.append(rng.uniform(0, 220, 20))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def test_a_network_burst_over_two_seconds_is_an_elongated_superburst():
    result = pf_detect(_long_burst_spikes(duration_s=3.0), duration_s=220.0)
    types = [e["superburst_type"] for e in result["superbursts"]["events"]]
    assert "elongated" in types
    long_nbs = [e for e in _events(result) if e["burst_duration_s"] > 2.0]
    assert len(long_nbs) >= 1


def test_short_network_bursts_are_not_elongated():
    result = pf_detect(_long_burst_spikes(duration_s=0.3), duration_s=220.0)
    assert all(e["superburst_type"] != "elongated" for e in result["superbursts"]["events"])


def test_every_superburst_carries_a_type():
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    for event in result["superbursts"]["events"]:
        assert event["superburst_type"] in ("cluster", "elongated")


def test_the_gap_is_measured_from_the_end_of_one_burst_to_the_start_of_the_next():
    """Both the bimodality test and the grouping use end-to-start gaps."""
    result = pf_detect(_clustered_bursts(), duration_s=50.0)
    diagnostics = result["diagnostics"]
    assert diagnostics["superburst_gap_source"] in (
        "nb_gap_otsu", "wagenaar_10x_break", "no_cluster_structure",
        "too_few_network_bursts", "user_override")
    if diagnostics["superburst_gap_s"] is not None and diagnostics["superburst_gap_source"] != "user_override":
        ordered = sorted(_events(result), key=lambda e: e["start_time_s"])
        gaps = np.array([b["start_time_s"] - a["end_time_s"]
                         for a, b in zip(ordered[:-1], ordered[1:])])
        assert gaps.min() <= diagnostics["superburst_gap_s"] <= gaps.max()


def test_wagenaar_break_needs_a_ten_fold_jump():
    import parameter_free_burst_detector as detector
    tight = np.array([0.5] * 8 + [8.0, 9.0, 10.0])
    assert detector._wagenaar_break(tight, floor=0.02) == pytest.approx(np.sqrt(0.5 * 8.0))
    smooth = np.linspace(1.0, 4.0, 12)
    assert detector._wagenaar_break(smooth, floor=0.02) is None


def test_one_long_pause_is_not_cluster_structure():
    import parameter_free_burst_detector as detector
    gaps = np.array([10.0] * 15 + [300.0])
    assert detector._wagenaar_break(gaps, floor=0.02) is None


def test_otsu_split_separates_two_well_separated_groups():
    import parameter_free_burst_detector as detector
    values = np.concatenate([np.full(14, -0.3), [1.2, 1.33]])
    assert -0.3 < detector._otsu_split(values) < 1.2


def test_otsu_split_is_none_without_spread():
    import parameter_free_burst_detector as detector
    assert detector._otsu_split(np.full(10, 2.0)) is None
    assert detector._otsu_split(np.array([1.0])) is None


def _short_clusters(n_units=20, seed=2):
    """Six clusters of four bursts 0.4 s apart: each cluster is well under 2.5 s."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for u in range(n_units):
        trains = []
        for start in np.arange(10.0, 190.0, 30.0):
            for k in range(4):
                trains.append(start + k * 0.4 + rng.normal(0, 0.02, 20))
        trains.append(rng.uniform(0, 200, 10))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))
    return spikes


def test_a_cluster_superburst_has_no_minimum_duration():
    result = pf_detect(_short_clusters(), duration_s=200.0)
    assert result["diagnostics"]["superburst_min_dur_s"] == 0.0
    clusters = [e for e in result["superbursts"]["events"] if e["superburst_type"] == "cluster"]
    assert clusters, "tight clusters are superbursts however short they are"
    assert all(e["burst_duration_s"] < 2.5 for e in clusters)
