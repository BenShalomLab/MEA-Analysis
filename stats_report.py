#!/usr/bin/env python3
# ==========================================================
# stats_report.py
#
# Group comparison across wells, and the HTML report of it.
#
# This is the step the pipeline was missing. Everything upstream produces
# numbers per well; nothing turned them into a statement about genotypes.
# Doing that correctly is mostly about respecting the design:
#
#   * The well is the replicate, nested in chip, nested in batch. Wells on one
#     chip share a dissection, a plating and a feeding history, so treating
#     them as independent samples inflates significance. Every model here
#     carries a random intercept for batch.
#   * Batch dominates. Mossink et al. 2021 found MEA batch explained 69% of
#     the variance in their parameters and astrocyte batch another 32%. The
#     variance partition is therefore reported next to every comparison, not
#     hidden in a supplement.
#   * Roughly thirty features are tested at once, so p-values are corrected
#     (Benjamini-Hochberg) and effect sizes with confidence intervals are
#     reported alongside them.
#   * Wells excluded by quality control are counted per genotype before any
#     of this, because unequal exclusion biases the comparison.
#
# Usage
# -----
#   python stats_report.py --table metrics/network_metrics_ALL.csv \
#       --out-dir reports/ --group-column sample_genotype
# ==========================================================

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qc_wells import evaluate_wells, exclusion_is_unbalanced, exclusion_summary
except ImportError:  # pragma: no cover
    from MEA_Analysis.IPNAnalysis.qc_wells import (
        evaluate_wells, exclusion_is_unbalanced, exclusion_summary,
    )

# Columns that identify a well rather than measure it.
IDENTIFIER_PREFIXES = ("sample_", "diag_")
IDENTIFIER_COLUMNS = {
    "project", "date", "chip", "run", "well", "output_dir", "data_dir",
    "error", "git_commit", "git_dirty", "detector", "qc_pass", "qc_reasons",
    "qc_n_rules_applied", "curation_applied",
}

DEFAULT_GROUP_COLUMN = "sample_genotype"
DEFAULT_BATCH_COLUMN = "sample_batch"
DEFAULT_COVARIATE = "sample_div"


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(table, min_finite_fraction=0.5, max_features=None):
    """Numeric measurement columns worth testing.

    Drops identifiers, near-constant columns and columns that are mostly
    missing: each one costs multiple-comparison power for every other feature,
    so testing metrics that cannot show anything is not free.
    """
    features = []
    for column in table.columns:
        if column in IDENTIFIER_COLUMNS or column.startswith(IDENTIFIER_PREFIXES):
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        finite = values.notna()
        if finite.mean() < min_finite_fraction:
            continue
        if values[finite].nunique() < 3:
            continue
        features.append(column)

    if max_features is not None:
        features = features[:max_features]
    return features


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def hedges_g(treatment, control):
    """Standardised mean difference with the small-sample correction.

    Returns (g, ci_low, ci_high). Reported because a p-value says only whether
    an effect is distinguishable from zero at this sample size; it says nothing
    about whether the effect is large enough to matter.
    """
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    treatment = treatment[np.isfinite(treatment)]
    control = control[np.isfinite(control)]

    n1, n2 = treatment.size, control.size
    if n1 < 2 or n2 < 2:
        return None, None, None

    pooled_variance = (
        (n1 - 1) * treatment.var(ddof=1) + (n2 - 1) * control.var(ddof=1)
    ) / (n1 + n2 - 2)
    if pooled_variance <= 0:
        return None, None, None

    d = (treatment.mean() - control.mean()) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = float(d * correction)

    standard_error = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, float(g - 1.96 * standard_error), float(g + 1.96 * standard_error)


# ---------------------------------------------------------------------------
# Mixed model
# ---------------------------------------------------------------------------

def fit_feature(table, feature, group_column, batch_column=None, covariate=None):
    """One feature against the grouping variable, respecting the nesting.

    A random intercept for batch is used whenever more than one batch is
    present. With a single batch the random effect is not identifiable and the
    model falls back to ordinary least squares, which is recorded in `model`
    so the reader knows the batch effect was not accounted for.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    columns = [feature, group_column]
    if batch_column and batch_column in table.columns:
        columns.append(batch_column)
    if covariate and covariate in table.columns:
        columns.append(covariate)

    data = table[columns].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    if covariate and covariate in data.columns:
        data[covariate] = pd.to_numeric(data[covariate], errors="coerce")
    data = data.dropna(subset=[feature, group_column])

    levels = list(pd.unique(data[group_column].astype(str)))
    if len(levels) < 2 or len(data) < 4:
        return []

    data = data.rename(columns={feature: "value", group_column: "group"})
    formula = "value ~ C(group)"
    if covariate and covariate in data.columns and data[covariate].notna().sum() > 2:
        data = data.dropna(subset=[covariate])
        if data[covariate].nunique() > 1:
            formula += f" + {covariate}"

    use_mixed = (
        batch_column in data.columns
        and data[batch_column].nunique() > 1
        and len(data) > data[batch_column].nunique() + 2
    )

    # Mixed models on small, unbalanced designs routinely warn about singular
    # covariance or boundary solutions. Emitting one warning per feature buries
    # the results, so they are captured and reported as a per-feature flag
    # instead: a fit that did not converge is visible in the output rather than
    # in scrollback.
    converged = None
    try:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if use_mixed:
                model = smf.mixedlm(formula, data, groups=data[batch_column]).fit(
                    reml=True, method="lbfgs"
                )
                model_name = "mixedlm_batch_random_intercept"
            else:
                model = smf.ols(formula, data).fit()
                model_name = "ols_single_batch"
        converged = bool(getattr(model, "converged", True)) and not any(
            "failed to converge" in str(w.message).lower() for w in caught
        )
    except Exception as exc:  # singular fits happen with small, unbalanced data
        return [{
            "feature": feature, "model": "failed", "error": str(exc),
            "n_wells": int(len(data)),
        }]

    reference = sorted(data["group"].astype(str).unique())[0]
    results = []
    for name in model.params.index:
        if not name.startswith("C(group)[T."):
            continue
        level = name.split("[T.", 1)[1].rstrip("]")
        treatment_values = data.loc[data["group"].astype(str) == level, "value"]
        control_values = data.loc[data["group"].astype(str) == reference, "value"]
        effect, ci_low, ci_high = hedges_g(treatment_values, control_values)

        results.append({
            "feature": feature,
            "model": model_name,
            "converged": converged,
            "reference": reference,
            "level": level,
            "coefficient": float(model.params[name]),
            "std_error": float(model.bse[name]),
            "p_value": float(model.pvalues[name]),
            "hedges_g": effect,
            "hedges_g_ci_low": ci_low,
            "hedges_g_ci_high": ci_high,
            "n_wells": int(len(data)),
            "n_reference": int(control_values.notna().sum()),
            "n_level": int(treatment_values.notna().sum()),
            "mean_reference": float(control_values.mean()),
            "mean_level": float(treatment_values.mean()),
        })
    return results


def compare_groups(table, features, group_column=DEFAULT_GROUP_COLUMN,
                   batch_column=DEFAULT_BATCH_COLUMN, covariate=DEFAULT_COVARIATE,
                   alpha=0.05):
    """Fit every feature and correct across them."""
    rows = []
    for feature in features:
        rows.extend(
            fit_feature(table, feature, group_column, batch_column, covariate)
        )
    results = pd.DataFrame(rows)
    if results.empty or "p_value" not in results.columns:
        return results

    from statsmodels.stats.multitest import multipletests

    testable = results["p_value"].notna()
    results["q_value"] = np.nan
    if testable.any():
        _, corrected, _, _ = multipletests(
            results.loc[testable, "p_value"], alpha=alpha, method="fdr_bh"
        )
        results.loc[testable, "q_value"] = corrected
    results["significant"] = results["q_value"] < alpha
    return results.sort_values("p_value", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variance partition
# ---------------------------------------------------------------------------

def _is_collinear_with_group(table, factor, group_column):
    """True when each level of `factor` sits entirely inside one group.

    Such a factor cannot be separated from the grouping variable: its
    intraclass correlation would just re-measure the genotype effect and read
    as a nuisance factor explaining the result. This happens whenever a line
    name doubles as a genotype label.
    """
    if group_column not in table.columns:
        return False
    pairs = table[[factor, group_column]].dropna()
    if pairs.empty:
        return False
    groups_per_level = pairs.groupby(factor)[group_column].nunique()
    return bool((groups_per_level <= 1).all())


def variance_partition(table, features, factors=("sample_batch", "sample_line", "chip"),
                       group_column=DEFAULT_GROUP_COLUMN):
    """Share of each feature's variance attributable to each nuisance factor.

    Computed as the intraclass correlation from a random-intercept-only model.
    A feature whose batch ICC is high is one where a genotype effect can be
    mimicked by which batch the wells came from.

    Factors that are collinear with the grouping variable are skipped: their
    ICC would be the genotype effect wearing a nuisance factor's name.
    """
    import statsmodels.formula.api as smf

    rows = []
    for factor in factors:
        if factor not in table.columns or table[factor].nunique() < 2:
            continue
        if _is_collinear_with_group(table, factor, group_column):
            rows.append({
                "feature": None, "factor": factor, "icc": None,
                "skipped": "collinear_with_group_column",
            })
            continue
        for feature in features:
            data = table[[feature, factor]].copy()
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
            data = data.dropna()
            if len(data) < 4 or data[factor].nunique() < 2:
                continue
            data = data.rename(columns={feature: "value"})
            try:
                # Same reasoning as fit_feature: boundary and singularity
                # warnings are expected here and would bury the output.
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm("value ~ 1", data, groups=data[factor]).fit(reml=True)
                between = float(model.cov_re.iloc[0, 0])
                within = float(model.scale)
                total = between + within
                if total <= 0:
                    continue
                rows.append({
                    "feature": feature,
                    "factor": factor,
                    "icc": between / total,
                    "n_groups": int(data[factor].nunique()),
                    "n_wells": int(len(data)),
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ordination and importance
# ---------------------------------------------------------------------------

def _feature_matrix(table, features):
    matrix = table[features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.loc[:, matrix.notna().mean() > 0.5]
    matrix = matrix.fillna(matrix.median())
    keep = matrix.std() > 0
    matrix = matrix.loc[:, keep]
    standardised = (matrix - matrix.mean()) / matrix.std()
    return standardised


def ordination(table, features, n_components=2):
    """PCA of the standardised feature matrix. Returns (coords, explained)."""
    from sklearn.decomposition import PCA

    matrix = _feature_matrix(table, features)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None, None

    model = PCA(n_components=min(n_components, matrix.shape[1], matrix.shape[0] - 1))
    coordinates = model.fit_transform(matrix.to_numpy())
    return coordinates, model.explained_variance_ratio_


def feature_importance(table, features, group_column=DEFAULT_GROUP_COLUMN,
                       n_repeats=20, seed=0):
    """Permutation importance from a random forest, cross-validated.

    Complements the per-feature models: they ask whether a feature differs on
    its own, this asks which features carry the genotype signal jointly. With
    few wells the accuracy is not the point and is reported only so an
    uninformative model is visible as one.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score

    matrix = _feature_matrix(table, features)
    labels = table.loc[matrix.index, group_column].astype(str)
    usable = labels.notna() & (labels != "nan")
    matrix, labels = matrix[usable], labels[usable]

    if matrix.shape[0] < 6 or labels.nunique() < 2:
        return None, None

    counts = labels.value_counts()
    folds = int(min(5, counts.min()))
    forest = RandomForestClassifier(n_estimators=300, random_state=seed)

    accuracy = None
    if folds >= 2:
        try:
            accuracy = float(
                cross_val_score(forest, matrix, labels, cv=folds).mean()
            )
        except Exception:
            accuracy = None

    forest.fit(matrix, labels)
    importance = permutation_importance(
        forest, matrix, labels, n_repeats=n_repeats, random_state=seed
    )
    ranked = pd.DataFrame({
        "feature": matrix.columns,
        "importance": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return ranked, accuracy


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_to_base64(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_feature_by_group(table, feature, group_column, covariate=None):
    """Strip plot of one feature, one point per well, split by group.

    Every well is drawn. Bar charts of means hide exactly the thing that
    matters here, which is how much wells within a group vary.
    """
    figure, axis = plt.subplots(figsize=(5, 3.6))
    groups = [g for g, _ in table.groupby(group_column, dropna=False)]
    rng = np.random.default_rng(0)

    for position, group in enumerate(groups):
        values = pd.to_numeric(
            table.loc[table[group_column] == group, feature], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, values.size)
        axis.scatter(position + jitter, values, s=26, alpha=0.75, edgecolors="none")
        axis.hlines(values.median(), position - 0.25, position + 0.25,
                    color="#2d3436", lw=2)

    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([str(g) for g in groups], rotation=20, ha="right")
    axis.set_ylabel(feature)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure


def plot_ordination(coordinates, labels, explained):
    figure, axis = plt.subplots(figsize=(5, 4.2))
    for group in pd.unique(labels):
        mask = labels == group
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1],
                     s=40, alpha=0.8, edgecolors="none", label=str(group))
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.0f}%)")
    if coordinates.shape[1] > 1:
        axis.set_ylabel(f"PC2 ({explained[1] * 100:.0f}%)")
    axis.legend(frameon=False, fontsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure


def plot_importance(ranked, top_n=15):
    top = ranked.head(top_n).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6, 0.28 * len(top) + 1.2))
    axis.barh(range(len(top)), top["importance"], xerr=top["importance_std"],
              color="#0984e3", alpha=0.85)
    axis.set_yticks(range(len(top)))
    axis.set_yticklabels(top["feature"], fontsize=8)
    axis.set_xlabel("permutation importance")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Helvetica, sans-serif; margin: 2rem auto;
        max-width: 60rem; color: #2d3436; line-height: 1.5; padding: 0 1rem; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; font-size: 0.82rem; width: 100%; }}
 th, td {{ border-bottom: 1px solid #dfe6e9; padding: 0.3rem 0.5rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .note {{ background: #f7f9fa; border-left: 3px solid #0984e3; padding: 0.6rem 0.9rem;
          font-size: 0.86rem; }}
 .warn {{ background: #fff5f5; border-left: 3px solid #d63031; padding: 0.6rem 0.9rem;
          font-size: 0.86rem; }}
 img {{ max-width: 100%; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
 .grid figure {{ margin: 0; flex: 1 1 22rem; }}
 figcaption {{ font-size: 0.78rem; color: #636e72; }}
</style></head><body>
<h1>{title}</h1>
<p class="note">{design}</p>
{warning}
<h2>Quality control</h2>
{qc_table}
<h2>Group comparison</h2>
<p class="note">{model_note}</p>
{results_table}
<h2>Feature distributions</h2>
<div class="grid">{feature_figures}</div>
<h2>Variance attributable to nuisance factors</h2>
<p class="note">{variance_note}</p>
{variance_table}
<h2>Multivariate</h2>
<div class="grid">{multivariate_figures}</div>
<hr><p style="font-size:0.78rem;color:#636e72">{footer}</p>
</body></html>
"""


def _table_html(frame, max_rows=40, float_format="{:.4g}"):
    if frame is None or len(frame) == 0:
        return "<p><em>Nothing to report.</em></p>"
    return frame.head(max_rows).to_html(
        index=False, float_format=lambda v: float_format.format(v), na_rep="—"
    )


def _figure_html(figure, caption):
    return (f'<figure><img src="data:image/png;base64,{_figure_to_base64(figure)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def build_report(table, group_column=DEFAULT_GROUP_COLUMN,
                 batch_column=DEFAULT_BATCH_COLUMN, covariate=DEFAULT_COVARIATE,
                 alpha=0.05, top_features=6, title="MEA group comparison"):
    """Run the whole analysis and return (html, results, qc, variance)."""
    qc = evaluate_wells(table)
    summary = exclusion_summary(qc, group_columns=(group_column,))
    passing = qc[qc["qc_pass"]].copy()

    features = select_features(passing)
    results = compare_groups(passing, features, group_column, batch_column,
                             covariate, alpha)
    variance = variance_partition(passing, features, group_column=group_column)

    warning = ""
    if exclusion_is_unbalanced(summary):
        warning = (
            '<p class="warn">Quality control removed a noticeably different share of '
            "wells from different groups. The surviving wells of the worst affected "
            "group are its healthiest, which biases every comparison below towards "
            "no effect. Check the exclusion table before reading the results.</p>"
        )

    n_batches = passing[batch_column].nunique() if batch_column in passing else 0
    model_note = (
        f"Linear mixed model per feature: the grouping variable as a fixed effect, "
        f"a random intercept for {batch_column} ({n_batches} levels), "
        f"{covariate} as a covariate where available. "
        f"p-values corrected across {len(results)} tests "
        f"(Benjamini-Hochberg, alpha {alpha}). Effect sizes are Hedges' g with 95% "
        f"confidence intervals."
    ) if n_batches > 1 else (
        "Only one batch is present, so no random intercept could be fitted and these "
        "are ordinary least squares fits. Batch effects are NOT accounted for; "
        "Mossink et al. 2021 found batch explained most of the variance in this assay, "
        "so treat any difference here as provisional until replicated across batches."
    )

    feature_figures = ""
    ordered = results[results["p_value"].notna()] if len(results) else results
    for feature in list(dict.fromkeys(ordered["feature"]))[:top_features]:
        figure = plot_feature_by_group(passing, feature, group_column)
        row = ordered[ordered["feature"] == feature].iloc[0]

        parts = [feature]
        q_value = row.get("q_value")
        if q_value is not None and np.isfinite(q_value):
            parts.append(f"q = {q_value:.3g}")
        effect = row.get("hedges_g")
        if effect is not None and np.isfinite(effect):
            parts.append(f"Hedges' g = {effect:.2f}")
        feature_figures += _figure_html(figure, ", ".join(parts))

    multivariate = ""
    coordinates, explained = ordination(passing, features)
    if coordinates is not None:
        labels = passing[group_column].astype(str).to_numpy()[: coordinates.shape[0]]
        multivariate += _figure_html(
            plot_ordination(coordinates, labels, explained),
            "Principal components of all features, one point per well.",
        )
    ranked, accuracy = feature_importance(passing, features, group_column)
    if ranked is not None:
        accuracy_text = f"cross-validated accuracy {accuracy:.2f}" if accuracy else "accuracy not estimable"
        multivariate += _figure_html(
            plot_importance(ranked),
            f"Random forest permutation importance ({accuracy_text}).",
        )

    variance_note = (
        "Intraclass correlation from a random-intercept-only model: the share of each "
        "feature's variance explained by a nuisance factor. High values mean a genotype "
        "effect in that feature could be produced by which batch or line the wells came "
        "from."
    )
    variance_display = variance
    if len(variance) and "icc" in variance.columns:
        variance_display = variance.sort_values("icc", ascending=False, na_position="last")
    skipped = (
        variance[variance.get("skipped").notna()]["factor"].tolist()
        if len(variance) and "skipped" in variance.columns else []
    )
    if skipped:
        variance_note += (
            f" Skipped as collinear with the grouping variable: {', '.join(skipped)}. "
            "Each of their levels sits inside a single group, so their variance share "
            "would be the group effect under another name."
        )

    html = _HTML_TEMPLATE.format(
        title=title,
        design=(
            f"{len(table)} wells, {len(passing)} passing quality control, "
            f"{passing[group_column].nunique() if group_column in passing else 0} groups, "
            f"{n_batches} batches. The well is the replicate and is nested in chip and "
            f"batch, so wells are not independent samples."
        ),
        warning=warning,
        qc_table=_table_html(summary),
        model_note=model_note,
        results_table=_table_html(results),
        feature_figures=feature_figures or "<p><em>No features to plot.</em></p>",
        variance_note=variance_note,
        variance_table=_table_html(variance_display),
        multivariate_figures=multivariate or "<p><em>Too few wells.</em></p>",
        footer="Generated by stats_report.py.",
    )
    return html, results, summary, variance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True,
                        help="CSV from collect_network_jsons.py")
    parser.add_argument("--out-dir", default="./reports")
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--batch-column", default=DEFAULT_BATCH_COLUMN)
    parser.add_argument("--covariate", default=DEFAULT_COVARIATE)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--title", default="MEA group comparison")
    args = parser.parse_args(argv)

    table = pd.read_csv(args.table)
    if args.group_column not in table.columns:
        parser.error(
            f"--group-column '{args.group_column}' is not in the table. "
            f"Populate it with a samples file (see docs/samples_schema.md); "
            f"without a grouping variable there is nothing to compare."
        )

    html, results, qc, variance = build_report(
        table, args.group_column, args.batch_column, args.covariate,
        args.alpha, title=args.title,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    results.to_csv(out_dir / "group_comparison.csv", index=False)
    qc.to_csv(out_dir / "qc_exclusions.csv", index=False)
    variance.to_csv(out_dir / "variance_partition.csv", index=False)

    n_significant = int(results["significant"].sum()) if "significant" in results else 0
    print(f"Wrote {out_dir / 'report.html'}")
    print(f"{len(results)} tests, {n_significant} significant after correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
