# ==========================================================
# qc_wells.py
#
# Well-level quality gates, applied before any group comparison.
#
# A well with four units and no network bursts still produces numbers, and
# those numbers will happily enter a group mean and move it. Excluding such
# wells is standard practice; what is not standard, and matters more, is
# checking *which* wells were excluded. If a knockout genotype loses twice as
# many wells to the activity gate as its control, the surviving knockout wells
# are the healthiest ones and every downstream comparison is biased towards no
# effect.
#
# So this module does two things: flag wells against thresholds, and report
# the exclusions broken down by experimental group.
#
# Default thresholds follow Mossink et al. 2021 (Stem Cell Reports), the
# reference design study for this assay: mean firing rate above 0.1 Hz, burst
# rate above 0.4 per minute, network burst rate above 1 per minute.
# ==========================================================

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GATES = {
    "min_units": 10,
    "min_mean_firing_rate_hz": 0.1,
    "min_burst_rate_per_min": 0.4,
    "min_network_burst_rate_per_min": 1.0,
    "max_bad_channel_fraction": 0.3,
}

# (gate name, column in the collector table, comparison, unit conversion)
# Conversion turns the stored unit into the unit the threshold is written in,
# so the thresholds stay readable as the per-minute figures the literature
# quotes.
_RULES = (
    ("min_units", "n_units", "min", 1.0),
    ("min_mean_firing_rate_hz", "ul_mean_firing_rate_hz_mean", "min", 1.0),
    ("min_burst_rate_per_min", "ul_mi_burst_rate_hz_mean", "min", 60.0),
    ("min_network_burst_rate_per_min", "nb_burst_rate_hz", "min", 60.0),
    ("max_bad_channel_fraction", "bad_channel_fraction", "max", 1.0),
)

_REASON_LABELS = {
    "min_units": "too few units",
    "min_mean_firing_rate_hz": "firing rate too low",
    "min_burst_rate_per_min": "burst rate too low",
    "min_network_burst_rate_per_min": "network burst rate too low",
    "max_bad_channel_fraction": "too many bad channels",
}


def evaluate_wells(table, gates=None):
    """Add `qc_pass`, `qc_reasons` and `qc_n_rules_applied` columns.

    Rules whose column is absent from the table are skipped rather than
    treated as failures, and the count of rules actually applied is recorded
    so a table missing a metric cannot silently pass everything.
    """
    thresholds = {**DEFAULT_GATES, **(gates or {})}
    frame = table.copy()

    applied = [
        (name, column, direction, scale)
        for name, column, direction, scale in _RULES
        if column in frame.columns and name in thresholds and thresholds[name] is not None
    ]

    reasons = [[] for _ in range(len(frame))]
    for name, column, direction, scale in applied:
        values = pd.to_numeric(frame[column], errors="coerce") * scale
        threshold = thresholds[name]
        if direction == "min":
            failed = values < threshold
        else:
            failed = values > threshold
        # A missing value is not a failure: the metric was never measured.
        failed = failed.fillna(False).to_numpy()
        for index in np.flatnonzero(failed):
            reasons[index].append(_REASON_LABELS.get(name, name))

    frame["qc_reasons"] = ["; ".join(r) for r in reasons]
    frame["qc_pass"] = [len(r) == 0 for r in reasons]
    frame["qc_n_rules_applied"] = len(applied)
    return frame


def exclusion_summary(table, group_columns=("sample_genotype",)):
    """Wells kept and dropped per experimental group, with the reasons.

    Read this before the statistics, not after. Unequal exclusion between
    genotypes is itself a result, and it biases every comparison that follows.
    """
    frame = table if "qc_pass" in table.columns else evaluate_wells(table)
    present = [c for c in group_columns if c in frame.columns]

    if not present:
        grouped = [((("all wells"),), frame)]
        index_names = ["group"]
    else:
        grouped = list(frame.groupby(present, dropna=False))
        index_names = present

    rows = []
    for key, group in grouped:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(index_names, key))
        n_total = len(group)
        n_pass = int(group["qc_pass"].sum())
        row.update({
            "n_wells": n_total,
            "n_pass": n_pass,
            "n_excluded": n_total - n_pass,
            "fraction_excluded": (n_total - n_pass) / n_total if n_total else None,
        })
        for label in _REASON_LABELS.values():
            row[f"excluded_{label.replace(' ', '_')}"] = int(
                group["qc_reasons"].str.contains(label, regex=False).sum()
            )
        rows.append(row)

    return pd.DataFrame(rows)


def exclusion_is_unbalanced(summary, max_difference=0.2):
    """True when exclusion rates differ enough between groups to worry about.

    Deliberately a blunt flag rather than a test: with a handful of groups
    there is no power for a formal one, and the point is to make a human look.
    """
    fractions = pd.to_numeric(summary.get("fraction_excluded"), errors="coerce").dropna()
    if fractions.size < 2:
        return False
    return bool((fractions.max() - fractions.min()) > max_difference)
