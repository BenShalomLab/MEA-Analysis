"""Tests for experimental metadata lookup (Epic B).

The sample table is the only place genotype, line, batch and plating date
enter the pipeline, so a silent mismatch here mislabels a whole experiment.
These tests pin the matching rules, the date handling and the failure modes.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from samples import (
    compute_div,
    load_samples,
    lookup_sample,
    parse_date,
    resolve_sample,
)


@pytest.fixture
def samples_csv(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "project,chip,well,genotype,line,prep_type,batch,plating_date\n"
        # Chip-wide default, then a per-well exception on the same chip.
        "CDKL5,16657,,WT,C57BL6,dissociated_mouse,B12,2025-01-15\n"
        "CDKL5,16657,well003,KO,Cdkl5_KO_3,dissociated_mouse,B12,2025-01-15\n"
        "CDKL5,16658,,KO,Cdkl5_KO_3,dissociated_mouse,B12,2025-01-15\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("2025-01-30", (2025, 1, 30)),
        ("2025/01/30", (2025, 1, 30)),
        ("2025_01_30", (2025, 1, 30)),
        ("20250130", (2025, 1, 30)),
        ("250130", (2025, 1, 30)),
        ("30-01-2025", (2025, 1, 30)),
        ("2025-01-30_plate2", (2025, 1, 30)),
    ],
)
def test_parse_date_accepts_the_formats_the_lab_uses(text, expected):
    parsed = parse_date(text)
    assert (parsed.year, parsed.month, parsed.day) == expected


def test_digit_runs_are_read_by_length_not_by_list_order():
    assert parse_date("20250130").year == 2025
    assert parse_date("250130").year == 2025


@pytest.mark.parametrize("text", ["", None, "not a date", "well000", "20259999"])
def test_unparseable_values_are_none_rather_than_a_guess(text):
    assert parse_date(text) is None


def test_div_is_days_between_plating_and_recording():
    assert compute_div("2025-01-01", "2025-01-29") == 28


def test_div_is_none_when_either_date_is_unknown():
    assert compute_div(None, "2025-01-29") is None
    assert compute_div("2025-01-01", None) is None


def test_negative_div_is_reported_not_suppressed():
    """A recording before its own plating date means the table is wrong;
    hiding it would let the error reach the statistics."""
    assert compute_div("2025-02-01", "2025-01-01") == -31


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_blank_key_cells_are_wildcards(samples_csv):
    rows, error = load_samples(samples_csv)
    assert error is None

    row, matched_on, _ = lookup_sample(rows, project="CDKL5", chip="16657", well="well000")
    assert row["genotype"] == "WT"
    assert set(matched_on) == {"project", "chip"}


def test_the_most_specific_row_wins(samples_csv):
    rows, _ = load_samples(samples_csv)
    row, matched_on, n_best = lookup_sample(rows, project="CDKL5", chip="16657", well="well003")
    assert row["genotype"] == "KO"
    assert "well" in matched_on
    assert n_best == 1


@pytest.mark.parametrize("well", ["well003", "003", "3", "WELL003"])
def test_well_spellings_are_equivalent(samples_csv, well):
    rows, _ = load_samples(samples_csv)
    row, _, _ = lookup_sample(rows, project="CDKL5", chip="16657", well=well)
    assert row["genotype"] == "KO"


def test_chip_ids_compare_numerically_despite_zero_padding(samples_csv):
    rows, _ = load_samples(samples_csv)
    row, _, _ = lookup_sample(rows, project="CDKL5", chip="016657", well="well000")
    assert row["genotype"] == "WT"


def test_a_key_the_pipeline_cannot_infer_blocks_rows_that_specify_it(samples_csv):
    """Deliberate: a row naming a project only applies to recordings known to
    be from that project. Matching it on an unknown value would attach
    metadata that nothing verified."""
    rows, _ = load_samples(samples_csv)
    row, _, _ = lookup_sample(rows, project=None, chip="16657", well="well000")
    assert row is None


def test_no_match_returns_nothing_rather_than_the_first_row(samples_csv):
    rows, _ = load_samples(samples_csv)
    row, matched_on, n_best = lookup_sample(rows, project="CDKL5", chip="99999", well="well000")
    assert row is None and matched_on == [] and n_best == 0


def test_equally_specific_rows_are_counted(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text(
        "chip,well,genotype\n16657,well000,WT\n16657,well000,KO\n", encoding="utf-8"
    )
    rows, _ = load_samples(path)
    row, _, n_best = lookup_sample(rows, chip="16657", well="well000")
    assert n_best == 2
    assert row["genotype"] == "WT"  # deterministic: first wins


def test_date_key_matches_across_formats(tmp_path):
    path = tmp_path / "bydate.csv"
    path.write_text("date,genotype\n2025-01-30,WT\n", encoding="utf-8")
    rows, _ = load_samples(path)
    row, _, _ = lookup_sample(rows, date="250130")
    assert row["genotype"] == "WT"


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------

def test_headers_are_case_and_space_insensitive(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text("Chip,Plating Date,GENOTYPE\n16657,2025-01-15,WT\n", encoding="utf-8")
    rows, error = load_samples(path)
    assert error is None
    assert rows[0]["plating_date"] == "2025-01-15"
    assert rows[0]["genotype"] == "WT"


def test_unknown_columns_are_preserved(tmp_path):
    path = tmp_path / "extra.csv"
    path.write_text("chip,genotype,coating\n16657,WT,laminin-521\n", encoding="utf-8")
    record = resolve_sample(path, chip="16657", well="well000")
    assert record["coating"] == "laminin-521"


def test_tsv_is_read_as_tab_separated(tmp_path):
    path = tmp_path / "samples.tsv"
    path.write_text("chip\tgenotype\n16657\tWT\n", encoding="utf-8")
    rows, error = load_samples(path)
    assert error is None and rows[0]["genotype"] == "WT"


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "gappy.csv"
    path.write_text("chip,genotype\n16657,WT\n,\n\n", encoding="utf-8")
    rows, _ = load_samples(path)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# resolve_sample: the block written into every result
# ---------------------------------------------------------------------------

def test_resolved_record_carries_metadata_and_div(samples_csv):
    record = resolve_sample(
        samples_csv, project="CDKL5", date="2025-02-12", chip="16657", well="well000"
    )
    assert record["matched"] is True
    assert record["genotype"] == "WT"
    assert record["line"] == "C57BL6"
    assert record["prep_type"] == "dissociated_mouse"
    assert record["batch"] == "B12"
    assert record["div"] == 28
    assert record["div_source"] == "plating_date"


def test_explicit_div_column_overrides_the_derived_value(tmp_path):
    """Some experiments count from a thaw or differentiation date instead."""
    path = tmp_path / "withdiv.csv"
    path.write_text(
        "chip,genotype,plating_date,div\n16657,WT,2025-01-01,55\n", encoding="utf-8"
    )
    record = resolve_sample(path, chip="16657", date="2025-01-29", well="well000")
    assert record["div"] == 55
    assert record["div_source"] == "column"


def test_unmatched_well_is_flagged_not_silently_blank(samples_csv):
    record = resolve_sample(samples_csv, project="CDKL5", chip="99999", well="well000")
    assert record["matched"] is False
    assert record.get("genotype") is None
    assert record["samples_file"] == str(samples_csv)


def test_missing_file_degrades_instead_of_raising(tmp_path):
    """A batch run must not abort because the metadata file moved."""
    record = resolve_sample(tmp_path / "absent.csv", chip="16657", well="well000")
    assert record["matched"] is False
    assert "not found" in record["error"]


def test_no_samples_file_configured_is_not_an_error():
    record = resolve_sample(None, chip="16657", well="well000")
    assert record == {
        "matched": False,
        "samples_file": None,
        "matched_on": [],
        "div": None,
        "div_source": None,
    }


def test_resolve_sample_logs_the_warning_when_nothing_matches(samples_csv):
    messages = []

    class _Logger:
        def info(self, message, *args):
            messages.append(("info", message % args if args else message))

        def warning(self, message, *args):
            messages.append(("warning", message % args if args else message))

    resolve_sample(samples_csv, project="CDKL5", chip="99999",
                   well="well000", logger=_Logger())
    assert any(level == "warning" for level, _ in messages)
