"""The collector's --detector filter keeps one detector's rows and drops the
columns that are empty for it (including leftovers of pre-schema-5 files)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collect_network_jsons import to_dataframes

ROWS = [
    {"project": "P", "well": "well000", "detector": "parameter_free", "bf_burst_count": 3, "sb_burst_count": 1},
    {"project": "P", "well": "well000", "detector": "gaussian", "nb_burst_count": 2},
    {"project": "P", "well": "well001", "detector": None, "nb_count": 5},  # pre-schema-5 file
]


def test_detector_keeps_only_that_detectors_rows_and_columns():
    df = to_dataframes(ROWS, detector="parameter_free")["P"]
    assert list(df["detector"]) == ["parameter_free"]
    assert "bf_burst_count" in df.columns
    assert "nb_burst_count" not in df.columns  # gaussian-only column
    assert "nb_count" not in df.columns        # legacy column from the old-format file


def test_no_detector_keeps_every_row():
    assert len(to_dataframes(ROWS)["P"]) == 3


def test_unknown_detector_returns_nothing():
    assert to_dataframes(ROWS[:1], detector="gaussian") == {}



def test_distributions_are_whole_lists_and_their_statistics_are_not_written(tmp_path):
    import json
    from collect_network_jsons import extract_row
    well = tmp_path / "P" / "250101" / "M1" / "Network" / "000001" / "well000"
    well.mkdir(parents=True)
    events = [{"start_time_s": 1.0, "burst_duration_s": 0.5, "n_components": 2, "peak_hz": [1, 2]},
              {"start_time_s": 4.0, "burst_duration_s": 1.5, "n_components": 3}]
    (well / "network_results.json").write_text(json.dumps({
        "detector": "parameter_free", "n_units": 10,
        "spike_participation": {"n_spikes_total": 100, "n_spikes_in_network_bursts": 7,
                                "fraction_spikes_in_network_bursts": 0.07, "percent_random_spikes": 93.0},
        "network_bursts": {"events": events, "metrics": {
            "burst_count": 2, "burst_rate_hz": 0.1, "burst_duration_p95_s": 1.4,
            "burst_duration_s": {"mean": 1.0, "std": 0.7, "cv": 0.7},
            "ibi_s": {"mean": 3.0, "std": 0.0, "cv": 0.0}}},
        "superbursts": {"events": [], "metrics": {"burst_count": 0}}}))
    row = extract_row(well / "network_results.json")
    assert json.loads(row["nb_burst_duration_s"]) == [0.5, 1.5]
    assert json.loads(row["nb_start_time_s"]) == [1.0, 4.0]
    assert "nb_peak_hz" not in row                      # list-valued event fields are skipped
    assert [k for k in row if k.startswith("sb_")] == ["sb_burst_count"]  # no events: only the scalar
    assert sorted(k for k in row if k.startswith("nb_")) == [
        "nb_burst_count", "nb_burst_duration_s", "nb_burst_rate_hz",   # scalars kept, stats of the list dropped
        "nb_ibi_s_cv", "nb_ibi_s_mean", "nb_ibi_s_std",                # not an event field, so kept
        "nb_n_components", "nb_start_time_s"]
    assert row["sp_n_spikes_total"] == 100 and row["sp_percent_random_spikes"] == 93.0
