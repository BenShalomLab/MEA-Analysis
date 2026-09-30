"""Tests for the identifiers the collector writes into each row.

The run column held the literal string "Network" for every row, because the
run id was read one level above the well instead of two. Every downstream
grouping by run was therefore a single group.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collect_network_jsons import extract_row


def _well_dir(tmp_path, project="CDKL5", date="2025-01-30", chip="16657",
              run="run_001", assay="Network", well="well000"):
    """The output tree mirrors the input tree: the assay folder sits between
    the run and the well."""
    path = tmp_path / project / date / chip / run / assay / well
    path.mkdir(parents=True)
    return path


def _write(well_dir, payload):
    path = well_dir / "network_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_MINIMAL = {
    "n_units": 12,
    "network_bursts": {"events": [], "metrics": {"burst_count": 0, "burst_rate_hz": 0.0}},
    "diagnostics": {"schema_version": 3, "detector": "parameter_free"},
}


def test_run_is_read_from_the_path_above_the_assay_folder(tmp_path):
    row = extract_row(_write(_well_dir(tmp_path), _MINIMAL))
    assert row["run"] == "run_001"
    assert row["run"] != "Network"


def test_the_remaining_path_ids_are_unchanged(tmp_path):
    row = extract_row(_write(_well_dir(tmp_path), _MINIMAL))
    assert row["project"] == "CDKL5"
    assert row["date"] == "2025-01-30"
    assert row["chip"] == "16657"
    assert row["well"] == "well000"


def test_ids_recorded_in_the_json_override_the_path(tmp_path):
    """Directory names are a guess; the pipeline writes the ids it read from
    the recording itself."""
    payload = dict(_MINIMAL, project="REAL", date="2025-02-01",
                   chip_id="99999", run_id="run_042", well="well003")
    row = extract_row(_write(_well_dir(tmp_path), payload))

    assert row["project"] == "REAL"
    assert row["date"] == "2025-02-01"
    assert row["chip"] == "99999"
    assert row["run"] == "run_042"
    assert row["well"] == "well003"


@pytest.mark.parametrize("value", [None, ""])
def test_a_blank_json_id_falls_back_to_the_path(tmp_path, value):
    row = extract_row(_write(_well_dir(tmp_path), dict(_MINIMAL, run_id=value)))
    assert row["run"] == "run_001"


def test_schema_version_and_detector_reach_the_table(tmp_path):
    """Without these a reader cannot tell which code produced a row, and the
    two detectors report some keys differently."""
    row = extract_row(_write(_well_dir(tmp_path), _MINIMAL))
    assert row["schema_version"] == 3
    assert row["detector"] == "parameter_free"


def test_a_file_from_before_the_schema_was_versioned_reads_as_null(tmp_path):
    payload = {"n_units": 5, "diagnostics": {}}
    row = extract_row(_write(_well_dir(tmp_path), payload))
    assert row["schema_version"] is None


def test_an_unreadable_file_reports_the_error_instead_of_raising(tmp_path):
    well_dir = _well_dir(tmp_path)
    path = well_dir / "network_results.json"
    path.write_text("{not json", encoding="utf-8")

    row = extract_row(path)
    assert "error" in row
    assert row["run"] == "run_001"
