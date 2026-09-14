"""Tests for pairwise connectivity and network topology (Epic D).

Elephant is not installed here, so the STTC is checked against the definition
in Cutts & Eglen 2014 computed by hand, plus the analytic limits: identical
trains give 1, independent trains give ~0, and a shift past the window
destroys the coupling.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synchrony import (
    compute_synchrony,
    graph_metrics,
    null_sttc_threshold,
    spike_time_tiling_coefficient as sttc,
    sttc_matrix,
    tiling_fraction,
)


def _poisson(rate_hz=5.0, duration=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, duration, int(rate_hz * duration)))


def _synchronous_groups(n_per_group=8, duration=100.0, seed=0):
    """Two groups that burst at interleaved times: within-group coupling only."""
    rng = np.random.default_rng(seed)
    spikes = {}
    for group, offset in enumerate((0.0, 5.0)):
        for unit in range(n_per_group):
            trains = [
                centre + rng.uniform(-0.01, 0.01, 12)
                for centre in np.arange(2 + offset, duration - 2, 10.0)
            ]
            spikes[f"g{group}u{unit}"] = np.sort(np.concatenate(trains))
    return spikes


# ---------------------------------------------------------------------------
# Tiling fraction
# ---------------------------------------------------------------------------

def test_tiling_fraction_matches_the_analytic_value():
    # Five well-separated spikes, each covering 2*dt of a 10 s recording.
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert tiling_fraction(times, 0.01, 0, 10) == pytest.approx(5 * 0.02 / 10)


def test_overlapping_tiles_are_counted_once():
    # Spikes 10 ms apart with dt = 10 ms: tiles overlap.
    times = np.array([1.0, 1.01])
    assert tiling_fraction(times, 0.01, 0, 10) == pytest.approx(0.03 / 10)


def test_tiles_are_clipped_at_the_recording_edges():
    # A spike at t=0 has half its tile outside the recording.
    assert tiling_fraction(np.array([0.0]), 0.01, 0, 10) == pytest.approx(0.01 / 10)


def test_empty_train_covers_nothing():
    assert tiling_fraction(np.array([]), 0.01, 0, 10) == 0.0


# ---------------------------------------------------------------------------
# STTC
# ---------------------------------------------------------------------------

def test_identical_trains_give_one():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sttc(times, times, 0.01, 0, 10) == pytest.approx(1.0)


def test_a_shift_inside_the_window_still_gives_one():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sttc(times, times + 0.005, 0.01, 0, 10) == pytest.approx(1.0)


def test_a_shift_past_the_window_destroys_the_coupling():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(sttc(times, times + 0.5, 0.01, 0, 10)) < 0.05


def test_independent_trains_give_about_zero():
    assert abs(sttc(_poisson(seed=1), _poisson(seed=2), 0.01, 0, 100)) < 0.05


def test_sttc_matches_the_definition_computed_by_hand():
    times_a = np.array([1.0, 2.0])
    times_b = np.array([1.005])
    dt, start, stop = 0.01, 0.0, 10.0

    tiling_a = tiling_fraction(times_a, dt, start, stop)
    tiling_b = tiling_fraction(times_b, dt, start, stop)
    proportion_a = 0.5   # one of A's two spikes is within dt of B
    proportion_b = 1.0   # B's only spike is within dt of A
    expected = 0.5 * (
        (proportion_a - tiling_b) / (1 - proportion_a * tiling_b)
        + (proportion_b - tiling_a) / (1 - proportion_b * tiling_a)
    )

    assert sttc(times_a, times_b, dt, start, stop) == pytest.approx(expected)


def test_sttc_is_not_inflated_by_firing_rate():
    """The measure exists because rate-sensitive ones would report a
    hypoactive genotype as less connected for the wrong reason."""
    dense = _poisson(rate_hz=20.0, seed=3)
    sparse = _poisson(rate_hz=2.0, seed=4)
    assert abs(sttc(dense, sparse, 0.01, 0, 100)) < 0.05


def test_empty_train_gives_nan():
    assert np.isnan(sttc(np.array([]), np.array([1.0]), 0.01, 0, 10))


def test_matrix_is_symmetric_with_a_nan_diagonal():
    trains = [_poisson(seed=s) for s in range(4)]
    matrix = sttc_matrix(trains, 0.01, 0, 100)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.all(np.isnan(np.diag(matrix)))


# ---------------------------------------------------------------------------
# Null threshold
# ---------------------------------------------------------------------------

def test_null_threshold_is_near_zero_for_independent_trains():
    trains = [_poisson(seed=s) for s in range(6)]
    threshold, info = null_sttc_threshold(trains, 0.01, 0, 100, n_pairs=20, n_shifts=5)
    assert info["n_null_values"] > 0
    assert abs(threshold) < 0.2


def test_real_synchrony_exceeds_the_null_threshold():
    spikes = _synchronous_groups()
    trains = [np.asarray(v) for v in spikes.values()]
    threshold, _ = null_sttc_threshold(trains, 0.01, 0, 100, n_pairs=40, n_shifts=5)
    # Two units of the same group fire together every burst.
    within_group = sttc(trains[0], trains[1], 0.01, 0, 100)
    assert within_group > threshold


def test_null_threshold_is_none_with_fewer_than_two_trains():
    threshold, info = null_sttc_threshold([_poisson()], 0.01, 0, 100)
    assert threshold is None and info["n_null_values"] == 0


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

def _block_matrix(n_per_block=6, within=0.8, between=0.0):
    size = 2 * n_per_block
    matrix = np.full((size, size), between, dtype=float)
    matrix[:n_per_block, :n_per_block] = within
    matrix[n_per_block:, n_per_block:] = within
    np.fill_diagonal(matrix, np.nan)
    return matrix


def test_graph_metrics_recover_two_modules():
    metrics = graph_metrics(_block_matrix(), threshold=0.5)
    assert metrics["n_modules"] == 2
    assert metrics["modularity"] > 0.3
    assert metrics["n_components"] == 2
    assert metrics["largest_component_fraction"] == pytest.approx(0.5)


def test_a_fully_connected_matrix_has_density_one():
    size = 8
    matrix = np.full((size, size), 0.9)
    np.fill_diagonal(matrix, np.nan)
    metrics = graph_metrics(matrix, threshold=0.5)
    assert metrics["density"] == pytest.approx(1.0)
    assert metrics["mean_clustering"] == pytest.approx(1.0)
    assert metrics["characteristic_path_length"] == pytest.approx(1.0)


def test_a_threshold_above_every_value_leaves_no_edges():
    metrics = graph_metrics(_block_matrix(), threshold=0.99)
    assert metrics["n_edges"] == 0
    assert "mean_clustering" not in metrics


def test_clustering_is_normalised_against_random_graphs_of_the_same_size():
    """Raw clustering rises with density, so comparing wells with different
    unit counts would compare the unit counts."""
    metrics = graph_metrics(_block_matrix(), threshold=0.5, n_random=5)
    assert metrics["clustering_normalised"] > 1.0


def test_hub_fraction_is_reported():
    metrics = graph_metrics(_block_matrix(), threshold=0.5)
    assert 0.0 <= metrics["hub_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_compute_synchrony_separates_the_two_groups():
    result = compute_synchrony(
        _synchronous_groups(), duration_s=100.0,
        n_null_pairs=40, n_null_shifts=5, n_random_graphs=3,
    )
    assert set(result["windows"]) == {"10ms", "25ms"}

    window = result["windows"]["10ms"]
    assert window["window_s"] == pytest.approx(0.010)
    assert window["n_pairs"] == 16 * 15 // 2
    assert window["graph"]["n_modules"] == 2
    assert window["graph"]["modularity"] > 0.3


def test_matrices_are_returned_separately_from_the_summary():
    result = compute_synchrony(_synchronous_groups(n_per_group=3), duration_s=100.0,
                               n_null_pairs=10, n_null_shifts=3, n_random_graphs=2)
    assert set(result["matrices"]) == set(result["windows"])
    assert result["matrices"]["10ms"].shape == (6, 6)
    assert len(result["unit_ids"]) == 6


def test_subsampling_is_applied_and_reported():
    spikes = {f"u{i}": _poisson(rate_hz=1.0, seed=i) for i in range(12)}
    result = compute_synchrony(spikes, duration_s=100.0, max_units=5,
                               n_null_pairs=10, n_null_shifts=3, n_random_graphs=2)
    assert result["params"]["n_units_subsampled_to"] == 5
    assert result["matrices"]["10ms"].shape == (5, 5)


def test_fewer_than_two_active_units_is_not_an_error():
    result = compute_synchrony({"u0": _poisson(), "u1": np.array([])}, duration_s=100.0)
    assert result["windows"] == {}
    assert result["params"]["reason"] == "fewer_than_two_active_units"


def test_duration_source_is_recorded():
    spikes = _synchronous_groups(n_per_group=2)
    with_duration = compute_synchrony(spikes, duration_s=100.0, n_null_pairs=5,
                                      n_null_shifts=2, n_random_graphs=2)
    without = compute_synchrony(spikes, duration_s=None, n_null_pairs=5,
                                n_null_shifts=2, n_random_graphs=2)
    assert with_duration["params"]["duration_source"] == "recording"
    assert without["params"]["duration_source"] == "spike_span"
