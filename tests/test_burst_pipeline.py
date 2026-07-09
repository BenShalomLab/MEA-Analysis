"""
Tests for the burst-detection → plot pipeline.

Contract tests use inspect / AST to derive expected keys from the consuming
functions themselves, so renaming a field in either the detector or the
plotter breaks the relevant test without needing to update hardcoded strings.
"""

import ast
import inspect
import json
import sys
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parameter_free_burst_detector import compute_network_bursts
import helper_functions as helper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bursty_spike_times(n_units=10, n_bursts=5, seed=42):
    """Synthetic spike trains with clear network bursts."""
    rng = np.random.default_rng(seed)
    burst_centers = np.linspace(10, 110, n_bursts)
    spike_times = {}
    for u in range(n_units):
        spikes = [center + rng.uniform(-0.10, 0.10, 20) for center in burst_centers]
        spikes.append(rng.uniform(0, 120, 10))
        spike_times[f"u{u}"] = np.sort(np.concatenate(spikes))
    return spike_times


def _single_unit_spike_times(seed=0):
    rng = np.random.default_rng(seed)
    return {"u0": np.sort(rng.uniform(0, 60, 200))}


def _subscript_keys_for_var(func, var_name):
    """
    Return every string key accessed as `var_name["key"]` anywhere in func's source.
    Used to derive what an event-consumer reads without hardcoding those names.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == var_name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def _plot_clean_network_data_params():
    """
    Keyword parameters of plot_clean_network that are sourced from plot_data
    (excludes the axis argument and plot-style kwargs).
    """
    PLOT_STYLE = {"ylim", "use_twinx"}
    sig = inspect.signature(helper.plot_clean_network)
    return {
        name
        for name, _ in sig.parameters.items()
        if name != "ax" and name not in PLOT_STYLE
    }


def _detector_burst_levels(result):
    """Return the sub-dicts that represent burst event levels (have 'events' key)."""
    return {k: v for k, v in result.items() if isinstance(v, dict) and "events" in v}


# ---------------------------------------------------------------------------
# 1. plot_data keys ↔ plot_clean_network parameters
# ---------------------------------------------------------------------------

def test_plot_data_keys_match_plot_clean_network_signature():
    """
    Keys in plot_data must exactly match the data parameters of plot_clean_network.
    Derived via inspect — renaming either side breaks this test.
    """
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    assert "error" not in result, f"Detector returned error: {result}"

    plot_data_keys = set(result["plot_data"].keys())
    func_params    = _plot_clean_network_data_params()

    assert plot_data_keys == func_params, (
        f"Mismatch between detector plot_data keys and plot_clean_network parameters.\n"
        f"  Keys only in plot_data:        {plot_data_keys - func_params}\n"
        f"  Params missing from plot_data: {func_params - plot_data_keys}"
    )


def test_plot_clean_network_called_with_plot_data_no_error():
    """End-to-end: unpack plot_data directly into plot_clean_network."""
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    assert "error" not in result
    fig, ax = plt.subplots()
    try:
        ax_out, _ = helper.plot_clean_network(ax, **result["plot_data"], use_twinx=True)
        assert ax_out is ax
    finally:
        plt.close(fig)


def test_plot_clean_network_use_twinx_false():
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    fig, ax = plt.subplots()
    try:
        helper.plot_clean_network(ax, **result["plot_data"], use_twinx=False)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2. compute_network_bursts output structure (structural, no hardcoded keys)
# ---------------------------------------------------------------------------

def test_detector_burst_levels_have_events_and_metrics():
    """
    Every burst-level entry (those with an 'events' key) must also have a
    'metrics' dict and a list of events.  Catches restructuring that drops
    either sub-key.
    """
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    assert "error" not in result
    levels = _detector_burst_levels(result)
    assert len(levels) >= 1, "Detector returned no burst levels"
    for name, level in levels.items():
        assert isinstance(level["events"], list),  f"'{name}[events]' is not a list"
        assert "metrics" in level,                 f"'{name}' is missing 'metrics'"
        assert isinstance(level["metrics"], dict), f"'{name}[metrics]' is not a dict"


def test_detector_diagnostics_are_scalar_valued():
    """
    diagnostics must be a non-empty dict of scalar values (int/float/str/bool).
    Catches cases where an array sneaks into diagnostics and breaks JSON export.
    """
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    assert "error" not in result
    diag = result.get("diagnostics", {})
    assert isinstance(diag, dict) and diag, "diagnostics is missing or empty"
    for k, v in diag.items():
        assert isinstance(v, (int, float, str, bool)), (
            f"diagnostics['{k}'] is not a scalar: {type(v)}"
        )


def test_plot_data_required_signals_same_length():
    """
    The positional (required) parameters of plot_clean_network after 'ax' are
    the mandatory time-series arrays.  Derived via inspect — their lengths in
    plot_data must all match.
    """
    sig = inspect.signature(helper.plot_clean_network)
    required_array_params = [
        name for name, p in sig.parameters.items()
        if name != "ax" and p.default is inspect.Parameter.empty
    ]
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    pd = result["plot_data"]
    lengths = {k: len(pd[k]) for k in required_array_params if k in pd}
    assert len(set(lengths.values())) == 1, (
        f"Required signal arrays have inconsistent lengths: {lengths}"
    )


def test_detector_burst_events_have_all_keys_needed_by_mark_burst_hierarchy():
    """
    Parse mark_burst_hierarchy's source with ast to find every ev["key"] it
    reads, then verify all detector event levels supply those keys.
    No key names are hardcoded here.
    """
    needed = _subscript_keys_for_var(helper.mark_burst_hierarchy, "ev")
    assert needed, "AST found no ev['key'] accesses in mark_burst_hierarchy — check the parser"

    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    levels = _detector_burst_levels(result)
    assert levels, "Detector returned no burst levels"

    for level_name, level_data in levels.items():
        for ev in level_data["events"]:
            missing = needed - set(ev.keys())
            assert not missing, (
                f"Events in '{level_name}' are missing keys that "
                f"mark_burst_hierarchy reads: {missing}"
            )


# ---------------------------------------------------------------------------
# 3. Burst detection behaviour
# ---------------------------------------------------------------------------

def test_detector_finds_bursts_on_bursty_data():
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times(n_bursts=5))
    assert "error" not in result
    nb_events = next(
        v["events"] for k, v in result.items()
        if "network" in k and isinstance(v, dict) and "events" in v
    )
    assert len(nb_events) >= 1, "Expected at least one network burst on clearly bursty data"


def test_detector_burst_events_sorted_by_start_time():
    """Events in every burst level must be in ascending start-time order."""
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    for level_name, level in _detector_burst_levels(result).items():
        starts = [ev["start_time_s"] for ev in level["events"]]
        assert starts == sorted(starts), (
            f"'{level_name}' events are not sorted by start_time_s"
        )


def test_detector_participation_fraction_in_range():
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    # Locate the participation signal by finding the first required positional
    # param of plot_clean_network after the time vector.
    sig = inspect.signature(helper.plot_clean_network)
    required = [
        name for name, p in sig.parameters.items()
        if name != "ax" and p.default is inspect.Parameter.empty
    ]
    participation_key = required[1]  # second required param is participation
    sig_array = result["plot_data"][participation_key]
    assert np.all(sig_array >= 0),        f"{participation_key} contains negative values"
    assert np.all(sig_array <= 1.0 + 1e-9), f"{participation_key} exceeds 1.0"


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

def test_detector_empty_units():
    result = compute_network_bursts(SpikeTimes={})
    assert result == {"error": "no_units"}


def test_detector_all_empty_spike_arrays():
    result = compute_network_bursts(SpikeTimes={"u0": np.array([]), "u1": np.array([])})
    assert result == {"error": "no_spikes"}


def test_detector_single_unit_no_crash():
    result = compute_network_bursts(SpikeTimes=_single_unit_spike_times())
    assert isinstance(result, dict)
    assert "error" not in result


def test_detector_single_spike_per_unit():
    spike_times = {f"u{i}": np.array([float(i)]) for i in range(5)}
    result = compute_network_bursts(SpikeTimes=spike_times)
    assert isinstance(result, dict)


def test_detector_two_unit_minimum():
    rng = np.random.default_rng(7)
    spike_times = {
        "u0": np.sort(rng.uniform(0, 60, 100)),
        "u1": np.sort(rng.uniform(0, 60, 100)),
    }
    result = compute_network_bursts(SpikeTimes=spike_times)
    assert "error" not in result


# ---------------------------------------------------------------------------
# 5. recursive_clean (JSON serialisation helper)
# ---------------------------------------------------------------------------

def test_recursive_clean_numpy_scalars():
    obj = {"a": np.int64(3), "b": np.float32(1.5), "c": np.array([1, 2, 3])}
    cleaned = helper.recursive_clean(obj)
    assert isinstance(cleaned["a"], int)
    assert isinstance(cleaned["b"], float)
    assert isinstance(cleaned["c"], list)


def test_recursive_clean_nested():
    obj = {"outer": {"inner": np.int64(99)}}
    cleaned = helper.recursive_clean(obj)
    assert cleaned["outer"]["inner"] == 99
    assert isinstance(cleaned["outer"]["inner"], int)


def test_recursive_clean_json_serialisable():
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    result.pop("plot_data")
    result.pop("unit_stats")
    cleaned = helper.recursive_clean(result)
    json.dumps(cleaned)  # must not raise


# ---------------------------------------------------------------------------
# 6. mark_burst_hierarchy end-to-end contract
# ---------------------------------------------------------------------------

def test_mark_burst_hierarchy_accepts_detector_output():
    """
    Pass all burst event levels from the detector into mark_burst_hierarchy.
    Level names are derived from detector output, not hardcoded.
    If any ev["key"] access inside the function doesn't match a detector
    event field, this raises a KeyError and the test fails.
    """
    result = compute_network_bursts(SpikeTimes=_bursty_spike_times())
    event_levels = [
        v["events"]
        for v in result.values()
        if isinstance(v, dict) and "events" in v
    ]
    assert len(event_levels) >= 3, (
        f"Expected at least 3 burst levels, got {len(event_levels)}"
    )
    fig, (ax_r, ax_n) = plt.subplots(2, 1)
    ax_n.set_ylim(0, 1)
    try:
        helper.mark_burst_hierarchy(
            ax_raster=ax_r,
            ax_network=ax_n,
            burstlets=event_levels[0],
            network_bursts=event_levels[1],
            superbursts=event_levels[2],
        )
    finally:
        plt.close(fig)
