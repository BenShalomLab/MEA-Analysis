#!/usr/bin/env python3
# ==========================================================
# track_units.py
#
# Follow the same neurons across recordings of one well.
#
# Neurodevelopmental phenotypes are usually shifted trajectories rather than
# deficits at one age, so the same well is recorded repeatedly. Comparing
# those recordings well-by-well answers "did the population change"; matching
# individual units across them answers "did these neurons change", which is a
# different and stronger question, and the only way to tell a unit that
# matured from a unit that appeared.
#
# Method: match on the mean waveform, as UnitMatch does (van Beest et al.
# 2024, Nat Methods), because functional properties are exactly what changes
# between recordings and so cannot be used to establish identity. Candidate
# pairs are gated by distance on the array, scored by waveform similarity with
# a small allowance for temporal misalignment, and assigned one-to-one by
# optimal matching rather than greedily.
#
# This is deliberately a self-contained implementation rather than a call into
# UnitMatch itself. The repository's UnitMatch integration imports a clone from
# a hardcoded absolute path that exists on one machine, and it is wired for
# merging over-split units within a single recording, which is a different
# problem. The scoring here is simpler than UnitMatch's probabilistic model:
# it produces a similarity, not a match probability, so the threshold is a
# choice the user makes and sees rather than a calibrated posterior.
# ==========================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Two recordings days apart still show the same soma within a few tens of
# microns; a larger jump is a different neuron, however similar the waveform.
DEFAULT_MAX_DISTANCE_UM = 50.0

# Cosine similarity of normalised waveforms. 0.85 is deliberately strict: a
# false match silently fabricates a developmental trajectory for a neuron that
# does not exist.
DEFAULT_MIN_SIMILARITY = 0.85

# Samples of shift allowed when comparing waveforms, to absorb differences in
# where the sorter placed the trough.
DEFAULT_MAX_LAG_SAMPLES = 5


def _normalise(waveform):
    waveform = np.asarray(waveform, dtype=float).ravel()
    waveform = waveform - waveform.mean()
    norm = np.linalg.norm(waveform)
    return waveform / norm if norm > 0 else waveform


def waveform_similarity(first, second, max_lag=DEFAULT_MAX_LAG_SAMPLES):
    """Best cosine similarity over a small range of lags, in [-1, 1].

    Lag tolerance matters because the two recordings were sorted
    independently, so the same spike can sit a sample or two earlier or later
    in its template window.
    """
    first = _normalise(first)
    second = _normalise(second)
    if first.size == 0 or second.size == 0:
        return -1.0

    length = min(first.size, second.size)
    first, second = first[:length], second[:length]

    best = -1.0
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            a, b = first[-lag:], second[:length + lag]
        elif lag > 0:
            a, b = first[:length - lag], second[lag:]
        else:
            a, b = first, second
        if a.size < 3:
            continue
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 0:
            best = max(best, float(np.dot(a, b) / denominator))
    return best


def _distance(first, second):
    if first is None or second is None:
        return None
    return float(np.linalg.norm(np.asarray(first, dtype=float)
                                - np.asarray(second, dtype=float)))


def match_sessions(earlier, later, min_similarity=DEFAULT_MIN_SIMILARITY,
                   max_distance_um=DEFAULT_MAX_DISTANCE_UM,
                   max_lag=DEFAULT_MAX_LAG_SAMPLES):
    """One-to-one matches between two recordings.

    Each session maps unit id to {"template": 1-D waveform, "location": (x, y)}.
    Returns a list of {"from", "to", "similarity", "distance_um"}.

    Assignment is optimal over the whole set rather than greedy, so one
    ambiguous pair cannot cascade into a chain of wrong matches.
    """
    from scipy.optimize import linear_sum_assignment

    earlier_ids = list(earlier.keys())
    later_ids = list(later.keys())
    if not earlier_ids or not later_ids:
        return []

    similarity = np.full((len(earlier_ids), len(later_ids)), -1.0)
    distances = np.full_like(similarity, np.nan)

    for i, first_id in enumerate(earlier_ids):
        first = earlier[first_id]
        for j, second_id in enumerate(later_ids):
            second = later[second_id]
            separation = _distance(first.get("location"), second.get("location"))
            distances[i, j] = np.nan if separation is None else separation
            # Spatial gate first: it is cheap and rules out most pairs.
            if separation is not None and separation > max_distance_um:
                continue
            similarity[i, j] = waveform_similarity(
                first.get("template"), second.get("template"), max_lag=max_lag
            )

    feasible = similarity >= min_similarity
    if not feasible.any():
        return []

    # linear_sum_assignment minimises, and infeasible pairs are given a cost
    # high enough that they are only chosen when nothing else is left; those
    # are filtered out afterwards.
    cost = np.where(feasible, -similarity, 1e6)
    rows, columns = linear_sum_assignment(cost)

    matches = []
    for row, column in zip(rows, columns):
        if not feasible[row, column]:
            continue
        matches.append({
            "from": earlier_ids[row],
            "to": later_ids[column],
            "similarity": float(similarity[row, column]),
            "distance_um": (None if np.isnan(distances[row, column])
                            else float(distances[row, column])),
        })
    return matches


def build_tracks(sessions, session_names=None, **match_kwargs):
    """Chain pairwise matches across an ordered list of sessions.

    Sessions must already be in recording order. Matching is between
    consecutive sessions only: a unit absent from one recording and present
    again later starts a new track rather than being bridged, because a gap
    is exactly where a false match is most likely.

    Returns (assignments, summary). `assignments` is a list of
    {"session", "unit_id", "track_id", "similarity_to_previous"}.
    """
    if session_names is None:
        session_names = [f"session{i}" for i in range(len(sessions))]

    assignments = []
    track_of_unit = {}
    next_track = 0

    for unit_id in sessions[0] if sessions else []:
        track_of_unit[unit_id] = next_track
        assignments.append({
            "session": session_names[0], "unit_id": unit_id,
            "track_id": next_track, "similarity_to_previous": None,
        })
        next_track += 1

    all_matches = []
    for index in range(1, len(sessions)):
        matches = match_sessions(sessions[index - 1], sessions[index], **match_kwargs)
        all_matches.append(matches)
        carried = {m["to"]: m for m in matches}

        new_track_of_unit = {}
        for unit_id in sessions[index]:
            match = carried.get(unit_id)
            if match is not None and match["from"] in track_of_unit:
                track = track_of_unit[match["from"]]
                similarity = match["similarity"]
            else:
                track = next_track
                next_track += 1
                similarity = None
            new_track_of_unit[unit_id] = track
            assignments.append({
                "session": session_names[index], "unit_id": unit_id,
                "track_id": track, "similarity_to_previous": similarity,
            })
        track_of_unit = new_track_of_unit

    counts = {}
    for entry in assignments:
        counts[entry["track_id"]] = counts.get(entry["track_id"], 0) + 1
    n_sessions = len(sessions)

    summary = {
        "n_sessions": n_sessions,
        "n_units_per_session": [len(s) for s in sessions],
        "n_tracks": len(counts),
        "n_tracks_in_all_sessions": sum(1 for c in counts.values() if c == n_sessions),
        "n_matches_per_transition": [len(m) for m in all_matches],
        "mean_similarity": (
            float(np.mean([m["similarity"] for matches in all_matches for m in matches]))
            if any(all_matches) else None
        ),
    }
    if n_sessions and sessions[0]:
        summary["fraction_tracked_throughout"] = (
            summary["n_tracks_in_all_sessions"] / len(sessions[0])
        )
    return assignments, summary


# ---------------------------------------------------------------------------
# Loading sessions from pipeline output
# ---------------------------------------------------------------------------

def load_session(well_dir):
    """Templates and locations for one well output directory.

    Prefers the analyzer, which has both; falls back to raw_mean_templates.npy
    (written by --extract-rawsortedspikes), which has waveforms but no
    positions, in which case matching relies on waveform shape alone.
    """
    well_dir = Path(well_dir)

    analyzer_dir = well_dir / "analyzer_output"
    if analyzer_dir.exists():
        try:
            import spikeinterface.full as si

            analyzer = si.load_sorting_analyzer(analyzer_dir)
            templates = analyzer.get_extension("templates").get_data()
            try:
                locations = analyzer.get_extension("unit_locations").get_data()
            except Exception:
                locations = None

            session = {}
            for index, unit_id in enumerate(analyzer.unit_ids):
                template = np.asarray(templates[index], dtype=float)
                extremum = int(np.argmax(np.max(np.abs(template), axis=0)))
                session[str(unit_id)] = {
                    "template": template[:, extremum],
                    "location": (None if locations is None
                                 else tuple(float(v) for v in locations[index][:2])),
                }
            return session
        except Exception as exc:
            print(f"[warn] could not read analyzer in {well_dir}: {exc}")

    raw_path = well_dir / "raw_mean_templates.npy"
    if raw_path.exists():
        raw = np.load(raw_path, allow_pickle=True).item()
        return {
            str(unit_id): {"template": entry["raw_mean_template"], "location": None}
            for unit_id, entry in raw.items()
        }

    return {}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("well_dirs", nargs="+",
                        help="Well output directories, in recording order")
    parser.add_argument("--out", default="unit_tracking.csv")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--max-distance-um", type=float, default=DEFAULT_MAX_DISTANCE_UM)
    args = parser.parse_args(argv)

    sessions, names = [], []
    for well_dir in args.well_dirs:
        session = load_session(well_dir)
        if not session:
            print(f"[warn] no units loaded from {well_dir}; skipping")
            continue
        sessions.append(session)
        names.append(str(well_dir))

    if len(sessions) < 2:
        parser.error("need at least two readable recordings to track units across")

    assignments, summary = build_tracks(
        sessions, names,
        min_similarity=args.min_similarity,
        max_distance_um=args.max_distance_um,
    )

    import csv

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["session", "unit_id", "track_id", "similarity_to_previous"]
        )
        writer.writeheader()
        writer.writerows(assignments)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} and {summary_path}")
    print(f"{summary['n_tracks']} tracks across {summary['n_sessions']} recordings; "
          f"{summary['n_tracks_in_all_sessions']} present throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
