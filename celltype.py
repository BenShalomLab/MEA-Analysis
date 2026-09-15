# ==========================================================
# celltype.py
#
# Putative cell-type classification from spike waveform shape.
#
# The pipeline already computed template metrics and then threw them away.
# They carry the one cell-type signal an extracellular array can give: narrow,
# fast-repolarising spikes come predominantly from fast-spiking interneurons,
# broader ones from regular-spiking pyramidal cells. Excitation/inhibition
# balance is the dominant hypothesis in SHANK, MECP2, SCN2A, SYNGAP1 and FMR1
# models, and a shift in that balance is invisible in metrics pooled over all
# units.
#
# IMPORTANT on interpretation. These are *waveform* classes, not validated
# cell types. The narrow/broad split is a proxy whose correspondence to
# inhibitory and excitatory identity is imperfect, and in dissociated and
# iPSC-derived cultures it has only recently begun to be checked against
# ground truth (Hornauer et al. 2026, bioRxiv, using chemogenetic labelling
# and defined excitatory/inhibitory mixtures). Accordingly:
#   * classes are named fast_spiking / regular_spiking, never excitatory or
#     inhibitory;
#   * the split is applied only when the waveform distribution is actually
#     bimodal, which it often is not in young iPSC cultures. When it is not,
#     every unit is returned as unclassified rather than forced into a class.
# ==========================================================

from __future__ import annotations

import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None

# Trough-to-peak duration is named differently across SpikeInterface versions.
TROUGH_TO_PEAK_ALIASES = ("peak_to_valley", "peak_to_trough_duration")
HALF_WIDTH_ALIASES = ("half_width",)

# A two-component fit has to beat a one-component fit by at least this much
# BIC before the split is believed. 10 is the conventional "strong evidence"
# margin; without a gate a GMM will always return two clusters.
MIN_BIC_IMPROVEMENT = 10.0

# Sarle's bimodality coefficient exceeds this for a genuinely bimodal
# distribution (its value for a uniform distribution).
BIMODALITY_THRESHOLD = 0.555

# Below this many somatic units the mixture fit is not worth believing.
MIN_UNITS_FOR_SPLIT = 20


def _bimodality_coefficient(values):
    """Sarle's bimodality coefficient, with the small-sample correction."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 4:
        return np.nan
    centred = values - values.mean()
    std = centred.std()
    if std <= 0:
        return np.nan
    skew = float(np.mean(centred ** 3) / std ** 3)
    kurtosis = float(np.mean(centred ** 4) / std ** 4) - 3.0
    correction = 3.0 * ((n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float((skew ** 2 + 1) / (kurtosis + correction))


def _first_available(columns, aliases):
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _as_records(template_metrics):
    """Accept a pandas DataFrame or a dict of dicts; return (ids, records)."""
    if hasattr(template_metrics, "to_dict"):
        records = template_metrics.to_dict(orient="index")
    else:
        records = dict(template_metrics)
    return list(records.keys()), records


def _to_milliseconds(values):
    """Durations come back in seconds from some versions and ms from others.

    Trough-to-peak is 0.2-1.5 ms for real spikes, so the two conventions differ
    by three orders of magnitude and cannot be confused.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and np.median(np.abs(finite)) < 0.01:
        return values * 1000.0, "s"
    return values, "ms"


def somatic_polarity(template):
    """"somatic" when the negative peak dominates, else "non_somatic".

    A template whose positive peak is larger is recorded from a dendrite or an
    axon rather than near the soma (Wolff et al. 2025). Its width says nothing
    about the cell's firing type, so it must be kept out of the mixture fit
    rather than counted as a broad-spiking cell.
    """
    template = np.asarray(template, dtype=float)
    if template.size == 0 or not np.any(np.isfinite(template)):
        return "unknown"
    return "somatic" if abs(np.nanmin(template)) >= np.nanmax(template) else "non_somatic"


def classify_cell_types(template_metrics, templates=None, min_units=MIN_UNITS_FOR_SPLIT,
                        min_bic_improvement=MIN_BIC_IMPROVEMENT, seed=0):
    """Split units into putative fast- and regular-spiking classes.

    `template_metrics` is a DataFrame (or dict of dicts) indexed by unit id,
    as produced by SpikeInterface's template_metrics extension.
    `templates` optionally maps unit id to that unit's waveform on its
    extremum channel, used only to exclude non-somatic templates.

    Returns {"units": {unit_id: {...}}, "summary": {...}}. Every unit appears
    in "units", including the ones that could not be classified.
    """
    unit_ids, records = _as_records(template_metrics)
    summary = {
        "n_units": len(unit_ids),
        "classified": False,
        "reason": None,
    }
    per_unit = {
        unit_id: {"cell_class": "unclassified", "cell_class_probability": None,
                  "waveform_polarity": None}
        for unit_id in unit_ids
    }

    if not unit_ids:
        summary["reason"] = "no_units"
        return {"units": per_unit, "summary": summary}

    columns = set().union(*(set(r.keys()) for r in records.values()))
    trough_column = _first_available(columns, TROUGH_TO_PEAK_ALIASES)
    width_column = _first_available(columns, HALF_WIDTH_ALIASES)
    if trough_column is None:
        summary["reason"] = "no_trough_to_peak_metric"
        return {"units": per_unit, "summary": summary}
    summary["trough_to_peak_column"] = trough_column

    # Polarity: keep dendritic/axonal templates out of the fit.
    if templates:
        for unit_id in unit_ids:
            template = templates.get(unit_id)
            if template is None:
                continue
            polarity = somatic_polarity(template)
            per_unit[unit_id]["waveform_polarity"] = polarity
            if polarity == "non_somatic":
                per_unit[unit_id]["cell_class"] = "non_somatic"
    summary["n_non_somatic"] = sum(
        1 for v in per_unit.values() if v["cell_class"] == "non_somatic"
    )

    candidates = [
        unit_id for unit_id in unit_ids
        if per_unit[unit_id]["cell_class"] != "non_somatic"
    ]
    trough_raw = np.array(
        [records[unit_id].get(trough_column, np.nan) for unit_id in candidates], dtype=float
    )
    trough_ms, source_unit = _to_milliseconds(trough_raw)
    summary["duration_units_in_source"] = source_unit

    features = [trough_ms]
    if width_column is not None:
        width_ms, _ = _to_milliseconds(
            np.array([records[u].get(width_column, np.nan) for u in candidates], dtype=float)
        )
        features.append(width_ms)
        summary["half_width_column"] = width_column

    matrix = np.column_stack(features)
    usable = np.all(np.isfinite(matrix), axis=1) & (trough_ms > 0)
    usable_ids = [candidates[i] for i in np.flatnonzero(usable)]
    matrix = matrix[usable]
    summary["n_units_with_metrics"] = int(matrix.shape[0])

    if matrix.shape[0] < min_units:
        summary["reason"] = "too_few_units_with_metrics"
        return {"units": per_unit, "summary": summary}

    # Bimodality gate 1: the trough-to-peak distribution itself.
    bimodality = _bimodality_coefficient(matrix[:, 0])
    summary["bimodality_coefficient"] = None if np.isnan(bimodality) else float(bimodality)

    if GaussianMixture is None:
        summary["reason"] = "sklearn_unavailable"
        return {"units": per_unit, "summary": summary}

    standardised = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0
    )

    one = GaussianMixture(n_components=1, covariance_type="full", random_state=seed)
    two = GaussianMixture(n_components=2, covariance_type="full", random_state=seed, n_init=5)
    one.fit(standardised)
    two.fit(standardised)
    bic_improvement = float(one.bic(standardised) - two.bic(standardised))
    summary["bic_one_component"] = float(one.bic(standardised))
    summary["bic_two_components"] = float(two.bic(standardised))
    summary["bic_improvement"] = bic_improvement

    # Bimodality gate 2: the mixture has to earn the extra component.
    is_bimodal = (
        bic_improvement >= min_bic_improvement
        and not np.isnan(bimodality)
        and bimodality > BIMODALITY_THRESHOLD
    )
    summary["bimodal"] = bool(is_bimodal)
    if not is_bimodal:
        # Young iPSC cultures frequently look like this. Forcing a split here
        # would invent a cell-type difference between wells out of noise.
        summary["reason"] = "waveform_distribution_not_bimodal"
        return {"units": per_unit, "summary": summary}

    labels = two.predict(standardised)
    probabilities = two.predict_proba(standardised)

    # The component with the shorter trough-to-peak duration is the
    # fast-spiking one.
    mean_trough = [matrix[labels == c, 0].mean() for c in (0, 1)]
    fast_component = int(np.argmin(mean_trough))

    for index, unit_id in enumerate(usable_ids):
        component = int(labels[index])
        per_unit[unit_id]["cell_class"] = (
            "fast_spiking" if component == fast_component else "regular_spiking"
        )
        per_unit[unit_id]["cell_class_probability"] = float(probabilities[index, component])
        per_unit[unit_id]["trough_to_peak_ms"] = float(matrix[index, 0])
        if width_column is not None:
            per_unit[unit_id]["half_width_ms"] = float(matrix[index, 1])

    n_fast = sum(1 for v in per_unit.values() if v["cell_class"] == "fast_spiking")
    n_regular = sum(1 for v in per_unit.values() if v["cell_class"] == "regular_spiking")
    summary.update({
        "classified": True,
        "n_fast_spiking": n_fast,
        "n_regular_spiking": n_regular,
        "n_unclassified": sum(
            1 for v in per_unit.values() if v["cell_class"] == "unclassified"
        ),
        "fast_spiking_fraction": (
            float(n_fast / (n_fast + n_regular)) if (n_fast + n_regular) else None
        ),
        # The common waveform proxy for excitation/inhibition balance. It is a
        # proxy: see the module docstring before reading it as an E/I ratio.
        "regular_to_fast_ratio": (float(n_regular / n_fast) if n_fast else None),
        "fast_spiking_trough_to_peak_ms": float(mean_trough[fast_component]),
        "regular_spiking_trough_to_peak_ms": float(mean_trough[1 - fast_component]),
    })
    return {"units": per_unit, "summary": summary}


def summarise_by_class(per_unit_classes, per_unit_features, fields):
    """Mean/median/n of selected per-unit features, split by cell class.

    This is the point of classifying at all: a firing rate averaged over every
    unit hides a change confined to one population.
    """
    by_class: dict[str, dict] = {}
    for unit_id, info in per_unit_classes.items():
        cell_class = info.get("cell_class", "unclassified")
        by_class.setdefault(cell_class, {field: [] for field in fields})
        features = per_unit_features.get(unit_id) or {}
        for field in fields:
            value = features.get(field)
            if value is not None and np.isfinite(value):
                by_class[cell_class][field].append(float(value))

    summary = {}
    for cell_class, collected in by_class.items():
        entry = {"n_units": sum(1 for v in per_unit_classes.values()
                                if v.get("cell_class") == cell_class)}
        for field, values in collected.items():
            if values:
                entry[field] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
        summary[cell_class] = entry
    return summary
