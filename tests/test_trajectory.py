"""Tests for unit tracking and developmental trajectories (Epic F).

The capability being pinned is detecting a shift in timing. A knockout that
reaches the same activity two weeks late looks identical at every single
timepoint and differs only in the shape of the curve.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pd = pytest.importorskip("pandas")

from track_units import build_tracks, match_sessions, waveform_similarity
from trajectory import (
    compute_trajectories,
    trajectory_feature_table,
    well_trajectory,
)


def _spike(seed=0, n_samples=60):
    """A plausible extracellular waveform: sharp trough, slower rebound."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    width = 2.0 + rng.uniform(0, 2)
    return (-np.exp(-((t - 20) ** 2) / (2 * width ** 2))
            + 0.3 * np.exp(-((t - 30) ** 2) / (2 * 6.0 ** 2)))


def _session(unit_ids, noise=0.01, seed=0, pitch=30.0):
    rng = np.random.default_rng(seed)
    return {
        unit: {
            "template": _spike(int(unit.lstrip("u")) if unit.startswith("u") else hash(unit) % 50)
            + rng.normal(0, noise, 60),
            "location": (float(index * pitch), 0.0),
        }
        for index, unit in enumerate(unit_ids)
    }


# ---------------------------------------------------------------------------
# Waveform similarity
# ---------------------------------------------------------------------------

def test_a_waveform_matches_itself():
    assert waveform_similarity(_spike(1), _spike(1)) == pytest.approx(1.0)


def test_similarity_survives_a_small_misalignment():
    """The two recordings are sorted independently, so the same spike can sit
    a sample or two earlier in its template window."""
    waveform = _spike(2)
    shifted = np.roll(waveform, 3)
    assert waveform_similarity(waveform, shifted, max_lag=5) > 0.95


def test_a_large_shift_is_not_absorbed():
    waveform = _spike(2)
    assert waveform_similarity(waveform, np.roll(waveform, 3), max_lag=0) < 0.95


def test_different_waveforms_score_lower_than_matching_ones():
    assert waveform_similarity(_spike(1), _spike(1)) > waveform_similarity(_spike(1), -_spike(1))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_the_same_units_are_matched_across_two_recordings():
    units = [f"u{i}" for i in range(10)]
    first = _session(units, noise=0.01, seed=0)
    second = _session(units, noise=0.02, seed=1)

    matches = match_sessions(first, second)
    assert len(matches) == 10
    assert all(m["from"] == m["to"] for m in matches)


def test_units_too_far_apart_are_never_matched():
    """A soma does not move hundreds of microns between recordings, however
    similar the waveform."""
    first = {"u0": {"template": _spike(1), "location": (0.0, 0.0)}}
    second = {"u0": {"template": _spike(1), "location": (500.0, 0.0)}}

    assert match_sessions(first, second, max_distance_um=50) == []
    assert len(match_sessions(first, second, max_distance_um=1000)) == 1


def test_dissimilar_waveforms_are_not_matched():
    first = {"u0": {"template": _spike(1), "location": (0.0, 0.0)}}
    second = {"u0": {"template": -_spike(1), "location": (0.0, 0.0)}}
    assert match_sessions(first, second, min_similarity=0.85) == []


def test_matching_is_one_to_one():
    units = [f"u{i}" for i in range(6)]
    first = _session(units, seed=0)
    second = _session(units, seed=1)
    matches = match_sessions(first, second)

    assert len({m["from"] for m in matches}) == len(matches)
    assert len({m["to"] for m in matches}) == len(matches)


def test_an_empty_recording_matches_nothing():
    assert match_sessions({}, _session(["u0"])) == []


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def test_units_present_throughout_form_one_track_each():
    units = [f"u{i}" for i in range(12)]
    sessions = [_session(units, seed=s) for s in range(3)]

    assignments, summary = build_tracks(sessions, ["div14", "div21", "div28"])

    assert summary["n_tracks"] == 12
    assert summary["n_tracks_in_all_sessions"] == 12
    assert summary["fraction_tracked_throughout"] == pytest.approx(1.0)
    assert len(assignments) == 36


def test_units_that_appear_later_start_a_new_track():
    first = _session([f"u{i}" for i in range(8)], seed=0)
    second = _session([f"u{i}" for i in range(8)], seed=1)
    second["newcomer"] = {"template": _spike(99), "location": (900.0, 0.0)}

    _, summary = build_tracks([first, second])
    assert summary["n_tracks"] == 9
    assert summary["n_tracks_in_all_sessions"] == 8


def test_units_that_disappear_do_not_end_the_track_of_others():
    units = [f"u{i}" for i in range(10)]
    sessions = [
        _session(units, seed=0),
        _session(units[:6], seed=1),
        _session(units[:6], seed=2),
    ]
    _, summary = build_tracks(sessions)
    assert summary["n_units_per_session"] == [10, 6, 6]
    assert summary["n_tracks_in_all_sessions"] == 6


def test_a_gap_is_not_bridged():
    """A unit missing from the middle recording starts a new track rather
    than being reconnected: a gap is where a false match is most likely."""
    units = [f"u{i}" for i in range(5)]
    sessions = [
        _session(units, seed=0),
        _session(units[:4], seed=1),
        _session(units, seed=2),
    ]
    assignments, _ = build_tracks(sessions, ["a", "b", "c"])

    frame = pd.DataFrame(assignments)
    tracks_of_u4 = frame[frame["unit_id"] == "u4"]["track_id"].nunique()
    assert tracks_of_u4 == 2


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def test_a_rising_curve_gives_a_positive_slope():
    ages = np.array([7, 14, 21, 28], dtype=float)
    result = well_trajectory(ages, ages * 0.01, "feature")
    assert result["slope_per_day"] == pytest.approx(0.01)
    assert result["slope_r_squared"] == pytest.approx(1.0)


def test_onset_and_plateau_are_interpolated_not_quantised():
    """Recording schedules differ between experiments, so the answer must not
    be pinned to the days that happen to have been sampled."""
    ages = np.array([0.0, 5.0, 10.0])
    values = np.array([0.0, 0.5, 1.0])
    result = well_trajectory(ages, values, "feature")
    # 10% of the peak falls a fifth of the way into the first interval.
    assert result["div_at_onset"] == pytest.approx(1.0)
    # 90% falls four fifths of the way into the second.
    assert result["div_at_plateau"] == pytest.approx(9.0)


def test_too_few_timepoints_is_reported_rather_than_fitted():
    result = well_trajectory(np.array([7.0, 14.0]), np.array([1.0, 2.0]), "f")
    assert result["reason"] == "too_few_timepoints"
    assert result["slope_per_day"] is None


def test_a_delayed_genotype_is_detected_as_a_timing_shift():
    """The whole point: at every single timepoint the two can overlap, and
    only the fitted curve separates them."""
    rows = []
    for genotype, delay in (("WT", 0), ("KO", 7)):
        for chip in range(3):
            for div in (7, 14, 21, 28, 35):
                value = 1.0 / (1.0 + np.exp(-(div - 14 - delay) / 3.0))
                rows.append({
                    "project": "P", "chip": f"c{genotype}{chip}", "well": "well000",
                    "sample_div": div, "sample_genotype": genotype,
                    "nb_burst_rate_hz": value * 0.1,
                })
    table = pd.DataFrame(rows)

    trajectories = compute_trajectories(table, ["nb_burst_rate_hz"])
    means = trajectories.groupby("sample_genotype")[["div_at_onset", "div_at_plateau"]].mean()

    assert means.loc["KO", "div_at_onset"] - means.loc["WT", "div_at_onset"] == pytest.approx(7, abs=1.5)
    assert means.loc["KO", "div_at_plateau"] - means.loc["WT", "div_at_plateau"] == pytest.approx(7, abs=1.5)


def test_one_row_per_well_per_feature():
    rows = [
        {"project": "P", "chip": f"c{c}", "well": "well000", "sample_div": div,
         "sample_genotype": "WT", "a": float(div), "b": float(div * 2)}
        for c in range(2) for div in (7, 14, 21)
    ]
    trajectories = compute_trajectories(pd.DataFrame(rows), ["a", "b"])
    assert len(trajectories) == 4
    assert set(trajectories["feature"]) == {"a", "b"}


def test_experimental_metadata_is_carried_through():
    rows = [
        {"project": "P", "chip": "c0", "well": "well000", "sample_div": div,
         "sample_genotype": "KO", "sample_batch": "B1", "a": float(div)}
        for div in (7, 14, 21)
    ]
    trajectories = compute_trajectories(pd.DataFrame(rows), ["a"])
    assert trajectories.iloc[0]["sample_genotype"] == "KO"
    assert trajectories.iloc[0]["sample_batch"] == "B1"


def test_the_wide_table_is_shaped_for_the_group_comparison():
    rows = [
        {"project": "P", "chip": f"c{c}", "well": "well000", "sample_div": div,
         "sample_genotype": "WT" if c == 0 else "KO", "a": float(div)}
        for c in range(2) for div in (7, 14, 21)
    ]
    trajectories = compute_trajectories(pd.DataFrame(rows), ["a"])
    wide = trajectory_feature_table(trajectories, "slope_per_day")

    assert "a__slope_per_day" in wide.columns
    assert "sample_genotype" in wide.columns
    assert len(wide) == 2


def test_a_table_without_div_yields_nothing():
    table = pd.DataFrame([{"project": "P", "chip": "c0", "well": "w", "a": 1.0}])
    assert compute_trajectories(table, ["a"]).empty
