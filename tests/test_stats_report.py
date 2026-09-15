"""Tests for well quality gates and the group comparison (Epic G).

The design facts these pin down: excluded wells are counted per group, the
well is nested in batch so batch is a random effect, features are corrected
across the whole set, and a factor that duplicates the grouping variable is
not reported as explaining its variance.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pd = pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from qc_wells import (
    DEFAULT_GATES,
    evaluate_wells,
    exclusion_is_unbalanced,
    exclusion_summary,
)
from stats_report import (
    build_report,
    compare_groups,
    fit_feature,
    hedges_g,
    select_features,
    variance_partition,
)


def _experiment(effect=-0.5, n_batches=3, wells_per_group=8, seed=0,
                line_equals_genotype=True):
    """Two genotypes across several batches, with a real effect on two features."""
    rng = np.random.default_rng(seed)
    rows = []
    for batch_index in range(n_batches):
        batch = f"B{batch_index + 1}"
        batch_offset = rng.normal(0, 0.25)
        for genotype, size in (("WT", 0.0), ("KO", effect)):
            for well in range(wells_per_group):
                rows.append({
                    "project": "P", "date": "2025-01-01",
                    "chip": f"chip{batch_index}", "run": "r1", "well": f"w{well}",
                    "sample_genotype": genotype,
                    "sample_batch": batch,
                    "sample_div": 28,
                    "sample_line": genotype if line_equals_genotype else f"L{batch_index}",
                    "n_units": int(abs(rng.normal(60, 8))),
                    "ul_mean_firing_rate_hz_mean": float(abs(rng.normal(2.0, 0.4))),
                    "ul_mi_burst_rate_hz_mean": float(abs(rng.normal(0.05, 0.01))),
                    "nb_burst_rate_hz": float(abs(
                        rng.normal(0.10 + size * 0.05 + batch_offset * 0.05, 0.012)
                    )),
                    "nb_burst_duration_s_mean": float(abs(rng.normal(0.4, 0.06))),
                    "conn_25ms_mean": float(abs(rng.normal(0.30 + size * 0.08, 0.04))),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def test_a_healthy_well_passes_every_gate():
    frame = evaluate_wells(_experiment())
    assert frame["qc_pass"].all()
    assert frame["qc_n_rules_applied"].iloc[0] >= 3


def test_a_quiet_well_is_excluded_with_a_reason():
    table = _experiment()
    table.loc[0, "nb_burst_rate_hz"] = 0.001      # 0.06 per minute
    table.loc[0, "n_units"] = 3

    frame = evaluate_wells(table)
    assert not frame.loc[0, "qc_pass"]
    assert "too few units" in frame.loc[0, "qc_reasons"]
    assert "network burst rate too low" in frame.loc[0, "qc_reasons"]


def test_thresholds_are_read_in_the_units_they_are_written_in():
    """Rates are stored in Hz and the literature quotes them per minute."""
    table = _experiment()
    table.loc[0, "nb_burst_rate_hz"] = 0.9 / 60.0     # just under 1 per minute
    table.loc[1, "nb_burst_rate_hz"] = 1.1 / 60.0     # just over

    frame = evaluate_wells(table)
    assert not frame.loc[0, "qc_pass"]
    assert frame.loc[1, "qc_pass"]


def test_a_missing_metric_is_not_counted_as_a_failure():
    table = _experiment().drop(columns=["nb_burst_rate_hz"])
    frame = evaluate_wells(table)
    assert frame["qc_pass"].all()
    # and the reduced number of applied rules is recorded
    assert frame["qc_n_rules_applied"].iloc[0] < len(DEFAULT_GATES)


def test_missing_values_do_not_silently_exclude_a_well():
    table = _experiment()
    table.loc[0, "nb_burst_rate_hz"] = np.nan
    assert evaluate_wells(table).loc[0, "qc_pass"]


def test_exclusions_are_counted_per_group():
    """Unequal exclusion biases every comparison, so it has to be visible."""
    table = _experiment()
    knockouts = table.index[table["sample_genotype"] == "KO"][:6]
    table.loc[knockouts, "n_units"] = 2

    summary = exclusion_summary(evaluate_wells(table))
    knockout_row = summary[summary["sample_genotype"] == "KO"].iloc[0]
    wildtype_row = summary[summary["sample_genotype"] == "WT"].iloc[0]

    assert knockout_row["n_excluded"] == 6
    assert wildtype_row["n_excluded"] == 0
    assert knockout_row["excluded_too_few_units"] == 6


def test_unbalanced_exclusion_is_flagged():
    table = _experiment()
    knockouts = table.index[table["sample_genotype"] == "KO"][:12]
    table.loc[knockouts, "n_units"] = 2

    summary = exclusion_summary(evaluate_wells(table))
    assert exclusion_is_unbalanced(summary) is True


def test_balanced_exclusion_is_not_flagged():
    summary = exclusion_summary(evaluate_wells(_experiment()))
    assert exclusion_is_unbalanced(summary) is False


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def test_hedges_g_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 40)
    effect, low, high = hedges_g(values, values)
    assert effect == pytest.approx(0.0)
    assert low < 0 < high


def test_hedges_g_recovers_a_known_separation():
    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 1.0, 400)
    treatment = rng.normal(1.0, 1.0, 400)
    effect, low, high = hedges_g(treatment, control)
    assert effect == pytest.approx(1.0, abs=0.2)
    assert low < effect < high


def test_hedges_g_needs_two_samples_per_group():
    assert hedges_g([1.0], [2.0, 3.0]) == (None, None, None)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def test_identifier_columns_are_not_tested():
    features = select_features(_experiment())
    assert "sample_genotype" not in features
    assert "project" not in features
    assert "nb_burst_rate_hz" in features


def test_mostly_missing_columns_are_dropped():
    table = _experiment()
    table["mostly_missing"] = np.nan
    table.loc[:2, "mostly_missing"] = 1.0
    assert "mostly_missing" not in select_features(table)


def test_constant_columns_are_dropped():
    table = _experiment()
    table["always_the_same"] = 1.0
    assert "always_the_same" not in select_features(table)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_a_real_effect_is_detected_and_a_null_feature_is_not():
    table = _experiment()
    results = compare_groups(table, select_features(table))

    real = results[results["feature"] == "nb_burst_rate_hz"].iloc[0]
    null = results[results["feature"] == "nb_burst_duration_s_mean"].iloc[0]

    assert real["significant"]
    assert not null["significant"]
    assert abs(real["hedges_g"]) > abs(null["hedges_g"])


def test_batch_is_a_random_effect_when_there_is_more_than_one():
    table = _experiment(n_batches=3)
    rows = fit_feature(table, "nb_burst_rate_hz", "sample_genotype", "sample_batch", "sample_div")
    assert rows[0]["model"] == "mixedlm_batch_random_intercept"


def test_a_single_batch_falls_back_to_ols_and_says_so():
    """With one batch the random intercept is not identifiable. The fallback
    has to be visible, because batch effects are then unaccounted for."""
    table = _experiment(n_batches=1)
    rows = fit_feature(table, "nb_burst_rate_hz", "sample_genotype", "sample_batch", "sample_div")
    assert rows[0]["model"] == "ols_single_batch"


def test_convergence_is_reported_per_feature():
    table = _experiment()
    results = compare_groups(table, select_features(table))
    assert "converged" in results.columns
    assert results["converged"].notna().all()


def test_p_values_are_corrected_across_features():
    table = _experiment()
    results = compare_groups(table, select_features(table))
    assert (results["q_value"] >= results["p_value"] - 1e-12).all()
    assert results["q_value"].max() <= 1.0


def test_one_group_yields_no_comparison():
    table = _experiment()
    table["sample_genotype"] = "WT"
    assert fit_feature(table, "nb_burst_rate_hz", "sample_genotype", "sample_batch") == []


# ---------------------------------------------------------------------------
# Variance partition
# ---------------------------------------------------------------------------

def test_batch_variance_is_reported():
    table = _experiment()
    variance = variance_partition(table, ["nb_burst_rate_hz"], factors=("sample_batch",))
    row = variance.iloc[0]
    assert row["factor"] == "sample_batch"
    assert 0.0 <= row["icc"] <= 1.0


def test_a_factor_that_duplicates_the_grouping_variable_is_skipped():
    """Otherwise its variance share is the genotype effect under another
    name, and it reads as a nuisance factor explaining the result."""
    table = _experiment(line_equals_genotype=True)
    variance = variance_partition(table, ["nb_burst_rate_hz"], factors=("sample_line",))
    assert variance.iloc[0]["skipped"] == "collinear_with_group_column"


def test_an_independent_factor_is_not_skipped():
    table = _experiment(line_equals_genotype=False)
    variance = variance_partition(table, ["nb_burst_rate_hz"], factors=("sample_line",))
    assert "skipped" not in variance.columns or variance["skipped"].isna().all()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_runs_end_to_end_and_embeds_its_figures():
    html, results, qc, variance = build_report(_experiment())
    assert html.startswith("<!doctype html>")
    assert "data:image/png;base64," in html
    assert len(results) > 0
    assert len(qc) == 2


def test_report_warns_when_exclusion_is_unbalanced():
    table = _experiment()
    knockouts = table.index[table["sample_genotype"] == "KO"][:12]
    table.loc[knockouts, "n_units"] = 2

    html, _, _, _ = build_report(table)
    assert "different share of" in html


def test_report_says_when_batch_could_not_be_modelled():
    html, _, _, _ = build_report(_experiment(n_batches=1))
    assert "Only one batch" in html
