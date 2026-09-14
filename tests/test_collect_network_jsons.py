"""Tests for flattening network_results.json into the analysis table."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collect_network_jsons import extract_row, to_dataframes


def _write_result(root, *, project="CDKL5", date="2025-02-12", chip="16657",
                  run="001234", well="well000", sample=None, ids=True):
    """Create one well folder mirroring the pipeline's output layout."""
    well_dir = root / project / date / chip / run / "Network" / well
    well_dir.mkdir(parents=True)

    payload = {
        "n_units": 42,
        "detector": "parameter_free",
        "duration_s": 300.0,
        "fs": 20000.0,
        "network_bursts": {
            "events": [
                {"start_time_s": 1.0, "end_time_s": 2.0, "burst_duration_s": 1.0},
                {"start_time_s": 11.0, "end_time_s": 12.0, "burst_duration_s": 1.0},
            ],
            "metrics": {"burst_count": 2, "burst_rate_hz": 0.0067,
                        "burst_duration_s": {"mean": 1.0, "std": 0.0, "cv": 0.0}},
        },
        "burst_fragments": {"events": [], "metrics": {"burst_count": 0}},
        "superbursts": {"events": [], "metrics": {"burst_count": 0}},
        "diagnostics": {"n_units": 42, "bin_size_ms": 25.0},
        "curation": {"applied": True, "n_units_input": 60, "n_units_rejected": 18,
                     "rejected_by_reason": {"Low Presence": 12, "High Contam": 6}},
        "provenance": {"git_commit": "abc123", "git_dirty": False},
        "spike_participation": {
            "n_spikes_total": 1000, "n_spikes_in_network_bursts": 700,
            "fraction_spikes_in_network_bursts": 0.7, "percent_random_spikes": 30.0,
        },
        "unit_level": {
            "summary": {
                "n_units": 42,
                "n_bursting_units_maxinterval": 30,
                "mi_burst_rate_hz": {"n": 30, "mean": 0.12, "median": 0.11,
                                     "std": 0.03, "cv": 0.25},
            },
            "params": {"maxinterval": {"min_spikes": 5}},
        },
        "sample": sample if sample is not None else {
            "matched": True, "genotype": "KO", "line": "Cdkl5_KO_3",
            "prep_type": "dissociated_mouse", "batch": "B12", "div": 28,
            "div_source": "plating_date", "plating_date": "2025-01-15",
        },
    }
    if ids:
        payload.update({"project": project, "date": date, "chip_id": chip,
                        "run_id": run, "well": well})

    (well_dir / "network_results.json").write_text(json.dumps(payload))
    return well_dir / "network_results.json"


def test_run_id_comes_from_the_json_not_the_assay_folder(tmp_path):
    row = extract_row(_write_result(tmp_path))
    assert row["run"] == "001234"
    assert row["chip"] == "16657"
    assert row["project"] == "CDKL5"
    assert row["well"] == "well000"


def test_path_fallback_finds_the_run_id_above_the_assay_folder(tmp_path):
    """Older results predate the id fields; the layout must still parse.

    Reading the run one level above the well returned the literal string
    "Network" for every row, because the output tree mirrors the input tree
    and ends in an assay folder.
    """
    row = extract_row(_write_result(tmp_path, ids=False))
    assert row["run"] == "001234"
    assert row["run"] != "Network"


def test_sample_metadata_becomes_columns(tmp_path):
    row = extract_row(_write_result(tmp_path))
    assert row["sample_genotype"] == "KO"
    assert row["sample_div"] == 28
    assert row["sample_batch"] == "B12"
    assert row["sample_prep_type"] == "dissociated_mouse"
    assert row["sample_matched"] is True


def test_missing_sample_block_is_flagged_not_dropped(tmp_path):
    row = extract_row(_write_result(tmp_path, sample={}))
    assert row["sample_matched"] is False
    assert row["sample_genotype"] is None


def test_curation_and_provenance_reach_the_table(tmp_path):
    row = extract_row(_write_result(tmp_path))
    assert row["curation_n_units_rejected"] == 18
    assert row["curation_rejected_low_presence"] == 12
    assert row["git_commit"] == "abc123"
    assert row["detector"] == "parameter_free"
    assert row["duration_s"] == 300.0


def test_sample_columns_sit_with_the_ids_not_among_the_metrics(tmp_path):
    rows = [extract_row(_write_result(tmp_path))]
    frame = to_dataframes(rows)["ALL"]
    columns = list(frame.columns)
    assert columns.index("sample_genotype") < columns.index("nb_burst_count")


def test_unit_level_summary_becomes_ul_columns(tmp_path):
    row = extract_row(_write_result(tmp_path))
    assert row["ul_n_bursting_units_maxinterval"] == 30
    assert row["ul_mi_burst_rate_hz_mean"] == pytest.approx(0.12)
    assert row["ul_mi_burst_rate_hz_median"] == pytest.approx(0.11)
    assert row["ul_mi_burst_rate_hz_n"] == 30


def test_spike_participation_becomes_sp_columns(tmp_path):
    row = extract_row(_write_result(tmp_path))
    assert row["sp_percent_random_spikes"] == pytest.approx(30.0)
    assert row["sp_fraction_spikes_in_network_bursts"] == pytest.approx(0.7)


def test_unreadable_json_produces_an_error_row_not_a_crash(tmp_path):
    well_dir = tmp_path / "p" / "d" / "c" / "r" / "Network" / "well000"
    well_dir.mkdir(parents=True)
    bad = well_dir / "network_results.json"
    bad.write_text("{not json")
    row = extract_row(bad)
    assert "error" in row
