"""Tests for single-unit burst detection (Epic C).

Both detectors are checked against trains whose burst structure is known
exactly, so a regression shows up as a wrong count or a wrong duration rather
than as a plausible-looking number.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unit_bursts import (
    compute_unit_burst_features,
    logisi_bursts,
    logisi_threshold,
    maxinterval_bursts,
    unit_burst_features,
)


def _exact_bursts(n_bursts=5, spikes_per_burst=8, intra_isi=0.02, period=10.0):
    """Perfectly regular bursts: n_bursts x spikes_per_burst spikes."""
    return np.concatenate([
        index * period + np.arange(spikes_per_burst) * intra_isi
        for index in range(n_bursts)
    ])


def _jittered_bursts(seed=0, n_bursts=20, spikes_per_burst=10,
                     intra_isi=0.015, jitter=0.004, duration=100.0, n_noise=30):
    rng = np.random.default_rng(seed)
    trains = [
        center + np.cumsum(rng.normal(intra_isi, jitter, spikes_per_burst))
        for center in np.linspace(2, duration - 2, n_bursts)
    ]
    trains.append(rng.uniform(0, duration, n_noise))
    return np.sort(np.concatenate(trains))


def _poisson(rate_hz=2.0, duration=100.0, seed=1):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, duration, int(rate_hz * duration)))


# ---------------------------------------------------------------------------
# MaxInterval
# ---------------------------------------------------------------------------

def test_maxinterval_finds_every_burst_and_no_extras():
    bursts = maxinterval_bursts(_exact_bursts())
    assert len(bursts) == 5
    assert all(end - start + 1 == 8 for start, end in bursts)


def test_maxinterval_needs_the_minimum_spike_count():
    # Pairs of spikes 20 ms apart, 10 s apart from each other.
    times = np.concatenate([[i * 10.0, i * 10.0 + 0.02] for i in range(5)])
    assert maxinterval_bursts(times) == []
    assert len(maxinterval_bursts(times, min_spikes=2)) == 5


def test_maxinterval_merges_bursts_closer_than_the_minimum_interval():
    """Merging only bites when min_ibi_s exceeds max_isi_end_s; at or below it
    the gap would already have extended the burst during detection."""
    first = np.arange(5) * 0.02
    second = first[-1] + 0.25 + np.arange(5) * 0.02
    times = np.concatenate([first, second])

    # 250 ms gap is past the 200 ms end threshold: two bursts.
    assert len(maxinterval_bursts(times)) == 2
    # Raise the merge window past the gap and they become one.
    assert len(maxinterval_bursts(times, min_ibi_s=0.3)) == 1


def test_maxinterval_extends_through_a_decelerating_tail():
    """The end threshold is looser than the start threshold, so a burst that
    slows down is not cut in two."""
    times = np.array([0.0, 0.02, 0.04, 0.06, 0.21, 0.36])
    bursts = maxinterval_bursts(times, min_spikes=5)
    assert len(bursts) == 1
    assert bursts[0] == (0, 5)


def test_maxinterval_finds_nothing_in_poisson_firing():
    assert maxinterval_bursts(_poisson()) == []


# ---------------------------------------------------------------------------
# logISI
# ---------------------------------------------------------------------------

def test_logisi_threshold_is_derived_from_a_bimodal_distribution():
    threshold, source = logisi_threshold(_jittered_bursts())
    assert source == "antimode"
    # Between the intra-burst intervals (~15 ms) and the gaps between bursts.
    assert 0.02 < threshold <= 0.1


def test_logisi_falls_back_to_the_cutoff_without_a_clear_split():
    threshold, source = logisi_threshold(_poisson(rate_hz=20.0))
    assert source == "cutoff"
    assert threshold == pytest.approx(0.1)


def test_logisi_threshold_never_exceeds_the_intraburst_ceiling():
    threshold, _ = logisi_threshold(_jittered_bursts(intra_isi=0.05, jitter=0.01))
    assert threshold <= 0.1


def test_logisi_finds_every_burst_in_a_jittered_train():
    assert len(logisi_bursts(_jittered_bursts())) == 20


def test_logisi_needs_at_least_ten_intervals_to_fit_a_threshold():
    threshold, source = logisi_threshold(np.array([0.0, 0.01, 0.02]))
    assert source == "cutoff"


# ---------------------------------------------------------------------------
# Per-unit metrics
# ---------------------------------------------------------------------------

def test_metrics_match_a_train_with_known_structure():
    features = unit_burst_features(_exact_bursts(), duration_s=50.0)

    assert features["mi_n_bursts"] == 5
    assert features["mi_burst_rate_hz"] == pytest.approx(0.1)
    # 8 spikes at 20 ms spacing spans 7 intervals = 140 ms.
    assert features["mi_burst_duration_mean_s"] == pytest.approx(0.14)
    assert features["mi_spikes_per_burst_mean"] == pytest.approx(8.0)
    assert features["mi_intraburst_rate_hz"] == pytest.approx(8 / 0.14)
    assert features["mi_fraction_spikes_in_bursts"] == pytest.approx(1.0)
    assert features["mi_ibi_mean_s"] == pytest.approx(10.0)
    assert features["mi_ibi_cv"] == pytest.approx(0.0, abs=1e-9)
    assert features["mi_is_bursting"] is True


def test_rates_use_the_recording_duration():
    times = _exact_bursts()
    short = unit_burst_features(times, duration_s=50.0)["mi_burst_rate_hz"]
    long = unit_burst_features(times, duration_s=500.0)["mi_burst_rate_hz"]
    assert long == pytest.approx(short / 10)


def test_a_unit_without_bursts_reports_zero_not_null():
    features = unit_burst_features(_poisson(), duration_s=100.0)
    assert features["mi_n_bursts"] == 0
    assert features["mi_fraction_spikes_in_bursts"] == 0.0
    assert features["mi_burst_duration_mean_s"] is None
    assert features["mi_is_bursting"] is False


def test_two_bursts_are_not_enough_to_call_a_unit_bursting():
    """One interval gives no variability estimate."""
    features = unit_burst_features(_exact_bursts(n_bursts=2), duration_s=50.0)
    assert features["mi_n_bursts"] == 2
    assert features["mi_is_bursting"] is False


def test_the_two_detectors_agree_on_a_clean_bursting_unit():
    features = unit_burst_features(_jittered_bursts(), duration_s=100.0)
    assert features["mi_n_bursts"] == features["li_n_bursts"] == 20
    assert features["agreement_jaccard"] > 0.95


def test_agreement_is_null_when_neither_detector_finds_a_burst():
    features = unit_burst_features(_poisson(), duration_s=100.0)
    assert features["agreement_jaccard"] is None


# ---------------------------------------------------------------------------
# Whole-well summary
# ---------------------------------------------------------------------------

def test_summary_counts_bursting_units_per_method():
    spikes = {f"burst{i}": _jittered_bursts(seed=i) for i in range(4)}
    spikes.update({f"poisson{i}": _poisson(seed=100 + i) for i in range(3)})

    result = compute_unit_burst_features(spikes, duration_s=100.0)
    summary = result["summary"]

    assert summary["n_units"] == 7
    assert summary["n_bursting_units_maxinterval"] == 4
    assert summary["n_bursting_units_logisi"] == 4
    assert summary["duration_source"] == "recording"


def test_summary_statistics_ignore_units_with_undefined_values():
    spikes = {
        "bursting": _jittered_bursts(),
        "silent": _poisson(rate_hz=0.5, seed=7),
    }
    result = compute_unit_burst_features(spikes, duration_s=100.0)
    duration = result["summary"]["mi_burst_duration_mean_s"]
    # Only the bursting unit has a defined mean burst duration; averaging the
    # other in as a zero would halve it.
    assert duration["n"] == 1
    assert duration["mean"] > 0


def test_per_unit_table_has_one_row_per_non_empty_unit():
    spikes = {"a": _jittered_bursts(), "b": _poisson(), "empty": np.array([])}
    result = compute_unit_burst_features(spikes, duration_s=100.0)
    assert set(result["units"]) == {"a", "b"}


def test_duration_falls_back_to_the_spike_span_and_says_so():
    result = compute_unit_burst_features({"a": _exact_bursts()}, duration_s=None)
    assert result["summary"]["duration_source"] == "spike_span"


def test_parameters_are_recorded_with_the_results():
    result = compute_unit_burst_features(
        {"a": _exact_bursts()}, duration_s=50.0,
        maxinterval_params={"min_spikes": 3},
    )
    assert result["params"]["maxinterval"]["min_spikes"] == 3
    assert result["params"]["logisi"]["cutoff_s"] == pytest.approx(0.1)


def test_no_units_returns_an_empty_result_rather_than_raising():
    assert compute_unit_burst_features({}, duration_s=100.0)["summary"]["n_units"] == 0


# ---------------------------------------------------------------------------
# unit_stats.csv assembly
# ---------------------------------------------------------------------------

def test_unit_stats_frame_joins_both_sources_without_duplicate_columns():
    pytest.importorskip("spikeinterface")
    from mea_reports import ReportsMixin

    detector_stats = {
        "u0": {"mean_firing_rate_hz": 2.0, "cv_isi": 1.4, "is_bursty": True},
        "u1": {"mean_firing_rate_hz": 0.5, "cv_isi": 0.9, "is_bursty": False},
    }
    burst_features = {
        "u0": {"mean_firing_rate_hz": 2.0, "mi_n_bursts": 12, "li_n_bursts": 11},
        "u1": {"mean_firing_rate_hz": 0.5, "mi_n_bursts": 0, "li_n_bursts": 0},
    }

    frame = ReportsMixin._build_unit_stats_frame(detector_stats, burst_features)

    assert list(frame.index) == ["u0", "u1"]
    assert frame.index.name == "unit_id"
    assert list(frame.columns).count("mean_firing_rate_hz") == 1
    assert frame.loc["u0", "mi_n_bursts"] == 12
    assert frame.loc["u0", "cv_isi"] == pytest.approx(1.4)


def test_unit_stats_frame_works_with_only_one_source():
    pytest.importorskip("spikeinterface")
    from mea_reports import ReportsMixin

    only_bursts = ReportsMixin._build_unit_stats_frame({}, {"u0": {"mi_n_bursts": 3}})
    assert only_bursts.loc["u0", "mi_n_bursts"] == 3

    assert ReportsMixin._build_unit_stats_frame({}, {}) is None
