"""--summary mode: scalars, end-to-start IBI, fragment totals, diagnostics and spike
participation are kept; per-event lists become median/log_iqr/p90."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collect_network_jsons import SUMMARY_METRICS, extract_summary_row, to_dataframes


def _events(durations):
    return [{"start_time_s": 10.0 * i, "end_time_s": 10.0 * i + d, "peak_time_s": 10.0 * i + d / 2,
             "rise_time_s": d / 2, "decay_time_s": d / 2, "burst_duration_s": d,
             "participation_fraction": 0.5 + 0.01 * i, "coactive_fraction_peak": 0.2 + 0.01 * i,
             "burst_area_spikes_per_unit": 2.0 * d, "coactive_fraction_max": 0.3, "n_fragments": 2,
             "n_components": 1, "spikes_per_burst": 50} for i, d in enumerate(durations)]


def _write(tmp_path, nb_durations, name="well000"):
    well = tmp_path / "P" / "250101" / "M1" / "Network" / "000001" / name
    well.mkdir(parents=True)
    nb = _events(nb_durations)
    (well / "network_results.json").write_text(json.dumps({
        "detector": "parameter_free", "n_units": 40, "project": "P", "well": name,
        "spike_participation": {"n_spikes_total": 1000, "n_spikes_in_network_bursts": 400,
                                "fraction_spikes_in_network_bursts": 0.4, "percent_random_spikes": 60.0},
        "diagnostics": {"schema_version": 6, "recording_duration_s": 300.0, "bin_size_ms": 50.0,
                        "detection_threshold": 0.05, "min_units_abs": 8, "low_yield": False},
        "burst_fragments": {"events": _events([0.1, 0.2, 0.3]), "metrics": {
            "burst_count": 3, "burst_rate_hz": 0.01, "burst_rate_per_min": 0.6,
            "ifbi_gap_s": {"mean": 5.0, "std": 1.0, "cv": 0.2}, "ifbi_s": {"mean": 9.0, "std": 1.0, "cv": 0.1}}},
        "network_bursts": {"events": nb, "metrics": {
            "burst_count": len(nb), "burst_rate_hz": 0.02, "burst_rate_per_min": 1.2, "duty_cycle": 0.1,
            "ibi_gap_s": {"mean": 8.0, "std": 2.0, "cv": 0.25}, "ibi_s": {"mean": 10.0, "std": 2.0, "cv": 0.2}}},
        "superbursts": {"events": [], "metrics": {"burst_count": 0}},
    }))
    return well / "network_results.json"


def test_no_lists_and_no_dropped_fields(tmp_path):
    row = extract_summary_row(_write(tmp_path, [1.0, 2.0, 4.0, 8.0, 16.0]))
    assert not [k for k, v in row.items() if isinstance(v, str) and v.startswith("[")]
    for dropped in ("start_time_s", "end_time_s", "peak_time_s", "rise_time_s", "decay_time_s"):
        assert not [k for k in row if k.endswith(dropped)], dropped
    assert "nb_ibi_s_mean" not in row and "bf_ifbi_s_mean" not in row      # onset-to-onset dropped
    assert row["nb_ibi_gap_s_mean"] == 8.0 and row["nb_ibi_gap_s_cv"] == 0.25  # end-to-start kept
    assert row["bf_ifbi_gap_s_mean"] == 5.0


def test_kept_scalars_diagnostics_participation_and_fragment_counts(tmp_path):
    row = extract_summary_row(_write(tmp_path, [1.0, 2.0, 4.0, 8.0, 16.0]))
    assert row["nb_burst_count"] == 5 and row["nb_burst_rate_per_min"] == 1.2 and row["nb_duty_cycle"] == 0.1
    assert row["bf_burst_count"] == 3
    assert row["nb_n_fragments_total"] == 10          # 5 events x 2 fragments
    assert row["sb_n_fragments_total"] == 0           # tier present, no events
    assert row["diag_detection_threshold"] == 0.05 and row["diag_recording_duration_s"] == 300.0
    assert row["sp_percent_random_spikes"] == 60.0 and row["n_units"] == 40


def test_each_metric_becomes_median_log_iqr_p90(tmp_path):
    d = [1.0, 2.0, 4.0, 8.0, 16.0]
    row = extract_summary_row(_write(tmp_path, d))
    for m in SUMMARY_METRICS:
        for stat in ("median", "log_iqr", "p90"):
            assert f"nb_{m}_{stat}" in row
    assert row["nb_burst_duration_s_median"] == 4.0
    assert math.isclose(row["nb_burst_duration_s_log_iqr"], math.log(8.0) - math.log(2.0))
    assert math.isclose(row["nb_burst_duration_s_p90"], float(np.percentile(d, 90)))
    assert math.isclose(row["nb_burst_area_spikes_per_unit_median"], 8.0)


def test_undefined_statistics_are_nan_not_zero(tmp_path):
    row = extract_summary_row(_write(tmp_path, [2.0, 3.0]))      # n=2: log_iqr and p90 undefined
    assert row["nb_burst_duration_s_median"] == 2.5
    assert math.isnan(row["nb_burst_duration_s_log_iqr"]) and math.isnan(row["nb_burst_duration_s_p90"])
    sb = extract_summary_row(_write(tmp_path, [2.0, 3.0], name="well001"))
    assert math.isnan(sb["sb_burst_duration_s_median"])          # empty tier


def test_summary_rows_flow_through_the_dataframe_builder(tmp_path):
    rows = [extract_summary_row(_write(tmp_path, [1.0, 2.0, 4.0, 8.0, 16.0]))]
    df = to_dataframes(rows, detector="parameter_free")["P"]
    assert "nb_burst_duration_s_median" in df.columns
