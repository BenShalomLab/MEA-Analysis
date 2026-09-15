"""Tests for waveform-based cell-type classification (Epic E).

The load-bearing behaviour is the refusal to split: a mixture model will
always return two clusters, so the tests check that an unimodal population
comes back unclassified rather than divided into invented cell types.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pd = pytest.importorskip("pandas")

from celltype import (
    classify_cell_types,
    somatic_polarity,
    summarise_by_class,
)


def _metrics(n_fast=30, n_regular=70, seed=0, in_seconds=False,
             column="peak_to_valley"):
    """Two waveform populations: narrow fast-spiking, broad regular-spiking."""
    rng = np.random.default_rng(seed)
    trough_to_peak = np.concatenate([
        rng.normal(0.30, 0.05, n_fast),
        rng.normal(0.85, 0.12, n_regular),
    ])
    half_width = np.concatenate([
        rng.normal(0.15, 0.03, n_fast),
        rng.normal(0.40, 0.07, n_regular),
    ])
    if in_seconds:
        trough_to_peak = trough_to_peak / 1000.0
        half_width = half_width / 1000.0
    return pd.DataFrame(
        {column: trough_to_peak, "half_width": half_width},
        index=[f"u{i}" for i in range(n_fast + n_regular)],
    )


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------

def test_a_negative_going_template_is_somatic():
    assert somatic_polarity(np.array([-1.0, -5.0, 2.0])) == "somatic"


def test_a_positive_going_template_is_not_somatic():
    """Dendritic and axonal templates are broad for reasons unrelated to
    firing type, so they must not be counted as regular-spiking cells."""
    assert somatic_polarity(np.array([-1.0, 5.0, 2.0])) == "non_somatic"


def test_an_empty_template_gives_unknown():
    assert somatic_polarity(np.array([])) == "unknown"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_two_populations_are_recovered_exactly():
    result = classify_cell_types(_metrics(30, 70))
    summary = result["summary"]

    assert summary["classified"] is True
    assert summary["bimodal"] is True
    assert summary["n_fast_spiking"] == 30
    assert summary["n_regular_spiking"] == 70
    assert summary["fast_spiking_fraction"] == pytest.approx(0.3)
    assert summary["regular_to_fast_ratio"] == pytest.approx(70 / 30)


def test_the_fast_class_is_the_narrow_one():
    summary = classify_cell_types(_metrics())["summary"]
    assert (summary["fast_spiking_trough_to_peak_ms"]
            < summary["regular_spiking_trough_to_peak_ms"])


def test_a_unimodal_population_is_left_unclassified():
    """A mixture model always finds two clusters; without this gate the
    pipeline would invent a cell-type difference out of noise."""
    result = classify_cell_types(_metrics(n_fast=0, n_regular=100))
    summary = result["summary"]

    assert summary["classified"] is False
    assert summary["reason"] == "waveform_distribution_not_bimodal"
    assert all(u["cell_class"] == "unclassified" for u in result["units"].values())


def test_too_few_units_is_refused_rather_than_fitted():
    result = classify_cell_types(_metrics(3, 7))
    assert result["summary"]["reason"] == "too_few_units_with_metrics"


def test_durations_in_seconds_and_milliseconds_give_the_same_answer():
    in_ms = classify_cell_types(_metrics(30, 70))["summary"]
    in_s = classify_cell_types(_metrics(30, 70, in_seconds=True))["summary"]

    assert in_s["duration_units_in_source"] == "s"
    assert in_ms["duration_units_in_source"] == "ms"
    assert in_s["n_fast_spiking"] == in_ms["n_fast_spiking"]
    assert (in_s["fast_spiking_trough_to_peak_ms"]
            == pytest.approx(in_ms["fast_spiking_trough_to_peak_ms"]))


def test_the_newer_spikeinterface_column_name_is_accepted():
    """SpikeInterface renamed peak_to_valley to peak_to_trough_duration."""
    result = classify_cell_types(_metrics(column="peak_to_trough_duration"))
    assert result["summary"]["trough_to_peak_column"] == "peak_to_trough_duration"
    assert result["summary"]["classified"] is True


def test_missing_the_duration_metric_is_reported_not_guessed():
    frame = pd.DataFrame({"half_width": [0.2] * 40},
                         index=[f"u{i}" for i in range(40)])
    assert classify_cell_types(frame)["summary"]["reason"] == "no_trough_to_peak_metric"


def test_non_somatic_units_are_excluded_from_the_fit():
    metrics = _metrics(30, 70)
    templates = {
        f"u{i}": (np.array([-1.0, 5.0, 2.0]) if i % 10 == 0 else np.array([-1.0, -5.0, 2.0]))
        for i in range(100)
    }
    result = classify_cell_types(metrics, templates=templates)
    summary = result["summary"]

    assert summary["n_non_somatic"] == 10
    assert summary["n_fast_spiking"] + summary["n_regular_spiking"] == 90
    assert result["units"]["u0"]["cell_class"] == "non_somatic"


def test_every_unit_appears_in_the_output():
    metrics = _metrics(30, 70)
    result = classify_cell_types(metrics)
    assert set(result["units"]) == set(metrics.index)


def test_class_probabilities_are_reported():
    result = classify_cell_types(_metrics())
    probabilities = [
        u["cell_class_probability"] for u in result["units"].values()
        if u["cell_class_probability"] is not None
    ]
    assert probabilities
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_units_with_missing_metrics_are_skipped_not_crashed_on():
    metrics = _metrics(30, 70)
    metrics.iloc[0, 0] = np.nan
    result = classify_cell_types(metrics)
    assert result["summary"]["n_units_with_metrics"] == 99
    assert result["units"]["u0"]["cell_class"] == "unclassified"


def test_no_units_is_not_an_error():
    empty = pd.DataFrame({"peak_to_valley": []})
    assert classify_cell_types(empty)["summary"]["reason"] == "no_units"


# ---------------------------------------------------------------------------
# Per-class summaries
# ---------------------------------------------------------------------------

def test_features_are_summarised_separately_for_each_class():
    classes = {
        "u0": {"cell_class": "fast_spiking"},
        "u1": {"cell_class": "fast_spiking"},
        "u2": {"cell_class": "regular_spiking"},
    }
    features = {
        "u0": {"mean_firing_rate_hz": 10.0},
        "u1": {"mean_firing_rate_hz": 20.0},
        "u2": {"mean_firing_rate_hz": 1.0},
    }
    summary = summarise_by_class(classes, features, ("mean_firing_rate_hz",))

    assert summary["fast_spiking"]["mean_firing_rate_hz"]["mean"] == pytest.approx(15.0)
    assert summary["regular_spiking"]["mean_firing_rate_hz"]["mean"] == pytest.approx(1.0)
    assert summary["fast_spiking"]["n_units"] == 2


def test_missing_feature_values_are_dropped_from_the_class_mean():
    classes = {"u0": {"cell_class": "fast_spiking"}, "u1": {"cell_class": "fast_spiking"}}
    features = {"u0": {"mi_burst_rate_hz": 0.5}, "u1": {"mi_burst_rate_hz": None}}
    summary = summarise_by_class(classes, features, ("mi_burst_rate_hz",))

    assert summary["fast_spiking"]["mi_burst_rate_hz"]["n"] == 1
    assert summary["fast_spiking"]["n_units"] == 2


# ---------------------------------------------------------------------------
# Reports-side helpers
# ---------------------------------------------------------------------------

def _reports_harness():
    pytest.importorskip("spikeinterface")
    import types
    from mea_reports import ReportsMixin

    harness = types.SimpleNamespace()
    harness.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    return harness, ReportsMixin


def test_waveform_channel_label_maps_through_the_sparsity():
    """The sparse column index is per unit, so two units on the same
    electrode were previously titled with different channel numbers."""
    import types
    harness, ReportsMixin = _reports_harness()
    harness.analyzer = types.SimpleNamespace(
        sparsity=types.SimpleNamespace(
            unit_id_to_channel_ids={"u0": ["ch41", "ch42"], "u1": ["ch42", "ch77"]}
        ),
        channel_ids=["ch00", "ch01"],
    )
    label = types.MethodType(ReportsMixin._waveform_channel_label, harness)

    assert label("u0", 1) == "ch42"
    assert label("u1", 0) == "ch42"


def test_waveform_channel_label_falls_back_to_dense_channel_ids():
    import types
    harness, ReportsMixin = _reports_harness()
    harness.analyzer = types.SimpleNamespace(sparsity=None, channel_ids=["ch00", "ch01"])
    label = types.MethodType(ReportsMixin._waveform_channel_label, harness)
    assert label("u0", 1) == "ch01"


def test_waveform_channel_label_never_raises():
    import types
    harness, ReportsMixin = _reports_harness()
    harness.analyzer = None
    label = types.MethodType(ReportsMixin._waveform_channel_label, harness)
    assert label("u0", 3) == "idx3"
