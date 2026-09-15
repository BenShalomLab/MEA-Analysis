# ==========================================================
# trajectory.py
#
# Developmental trajectories: how a well's activity changes with age.
#
# Neurodevelopmental phenotypes are frequently shifts in timing rather than
# differences at any single age. A knockout culture that reaches the same
# network burst rate as its control two weeks later looks identical at the
# late timepoint and identical at the early one, and differs only in the shape
# of the curve between them. Comparing wells at one DIV cannot see that;
# comparing the fitted trajectories can.
#
# What is fitted here is deliberately simple and interpretable rather than a
# growth model: for each well and feature, the slope against DIV, the age at
# which activity first appears, and the age at which it stops rising. Those
# three describe the curve well enough to compare genotypes, and each survives
# the handful of timepoints a real experiment has. A sigmoid fit would look
# more sophisticated and would not be identifiable from five recordings.
# ==========================================================

from __future__ import annotations

import numpy as np
import pandas as pd

# Identify one culture followed over time. Well alone is not enough: well000
# exists on every chip.
DEFAULT_WELL_KEYS = ("project", "chip", "well")

DEFAULT_DIV_COLUMN = "sample_div"

# Fraction of a well's maximum at which a feature counts as having "arrived"
# and, separately, as having stopped rising.
ONSET_FRACTION = 0.1
PLATEAU_FRACTION = 0.9

# Below this many timepoints a slope is not worth fitting.
MIN_TIMEPOINTS = 3

# numpy renamed trapz to trapezoid in 2.0 and the repository pins 1.26, so
# both spellings have to work.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _fit_slope(ages, values):
    """Least-squares slope per day, with the fraction of variance explained."""
    if ages.size < 2 or np.ptp(ages) <= 0:
        return None, None
    slope, intercept = np.polyfit(ages, values, 1)
    predicted = slope * ages + intercept
    total = float(np.sum((values - values.mean()) ** 2))
    residual = float(np.sum((values - predicted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else None
    return float(slope), r_squared


def _crossing_age(ages, values, level):
    """Age at which the curve first reaches `level`, linearly interpolated.

    Interpolating rather than returning the first recorded age above the level
    keeps the answer from being quantised to the recording schedule, which is
    usually every few days and differs between experiments.
    """
    above = np.flatnonzero(values >= level)
    if above.size == 0:
        return None
    first = int(above[0])
    if first == 0:
        return float(ages[0])

    previous_age, previous_value = ages[first - 1], values[first - 1]
    current_age, current_value = ages[first], values[first]
    if current_value == previous_value:
        return float(current_age)
    fraction = (level - previous_value) / (current_value - previous_value)
    return float(previous_age + fraction * (current_age - previous_age))


def well_trajectory(ages, values, feature=""):
    """Trajectory descriptors for one well and one feature."""
    order = np.argsort(ages)
    ages = np.asarray(ages, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]

    finite = np.isfinite(ages) & np.isfinite(values)
    ages, values = ages[finite], values[finite]

    result = {
        "feature": feature,
        "n_timepoints": int(ages.size),
        "div_first": float(ages[0]) if ages.size else None,
        "div_last": float(ages[-1]) if ages.size else None,
        "value_first": float(values[0]) if values.size else None,
        "value_last": float(values[-1]) if values.size else None,
        "value_max": float(values.max()) if values.size else None,
        "slope_per_day": None,
        "slope_r_squared": None,
        "div_at_onset": None,
        "div_at_plateau": None,
        "auc_per_day": None,
    }
    if ages.size < MIN_TIMEPOINTS:
        result["reason"] = "too_few_timepoints"
        return result

    result["slope_per_day"], result["slope_r_squared"] = _fit_slope(ages, values)

    peak = values.max()
    if peak > 0:
        result["div_at_onset"] = _crossing_age(ages, values, ONSET_FRACTION * peak)
        result["div_at_plateau"] = _crossing_age(ages, values, PLATEAU_FRACTION * peak)

    # Area under the curve, divided by the span, so wells followed for
    # different lengths of time stay comparable.
    span = ages[-1] - ages[0]
    if span > 0:
        result["auc_per_day"] = float(_trapezoid(values, ages) / span)

    return result


def compute_trajectories(table, features, div_column=DEFAULT_DIV_COLUMN,
                         well_keys=DEFAULT_WELL_KEYS):
    """Trajectory descriptors for every well and feature in a collector table.

    Returns one row per well per feature. Wells with fewer than
    MIN_TIMEPOINTS recordings are reported with the reason rather than
    dropped, so a design that cannot support trajectory analysis is visible
    instead of silently producing an empty table.
    """
    keys = [k for k in well_keys if k in table.columns]
    if not keys or div_column not in table.columns:
        return pd.DataFrame()

    rows = []
    for key_values, group in table.groupby(keys, dropna=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        identity = dict(zip(keys, key_values))

        ages = pd.to_numeric(group[div_column], errors="coerce").to_numpy()
        for feature in features:
            if feature not in group.columns:
                continue
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy()
            entry = dict(identity)
            entry.update(well_trajectory(ages, values, feature))
            # Carry the experimental metadata so trajectories can be grouped.
            for column in group.columns:
                if column.startswith("sample_") and column != div_column:
                    unique = group[column].dropna().unique()
                    if unique.size == 1:
                        entry[column] = unique[0]
            rows.append(entry)

    return pd.DataFrame(rows)


def trajectory_feature_table(trajectories, statistic="slope_per_day"):
    """Pivot one trajectory statistic into a well-by-feature table.

    That shape is what the group comparison in stats_report.py expects, so a
    trajectory becomes just another set of per-well measurements to compare
    between genotypes.
    """
    if trajectories.empty or statistic not in trajectories.columns:
        return pd.DataFrame()

    identity = [c for c in trajectories.columns
                if c in DEFAULT_WELL_KEYS or c.startswith("sample_")]
    wide = trajectories.pivot_table(
        index=identity, columns="feature", values=statistic, aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        c: f"{c}__{statistic}" for c in wide.columns if c not in identity
    })
    return wide
