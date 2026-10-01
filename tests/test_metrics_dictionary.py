"""Tests that the metric dictionary describes what the detectors emit.

Without these the dictionary rots: a key gets added to a detector, nothing
documents its units, and a reader or an agent guesses. The coverage test below
fails on the next undocumented key rather than at analysis time.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import burst_common
from parameter_free_burst_detector import compute_network_bursts as pf_detect
from gaussianNetworkBursts import compute_network_bursts as gauss_detect

DICTIONARY_PATH = os.path.join(ROOT, "metrics.json")
TIERS = ("burst_fragments", "network_bursts", "superbursts")


@pytest.fixture(scope="module")
def dictionary():
    with open(DICTIONARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def results():
    """One synthetic well through both detectors, with merged bursts and
    superbursts present so every tier populates."""
    rng = np.random.default_rng(5)
    spikes = {}
    for u in range(15):
        trains = []
        for cluster_start in (10.0, 30.0):
            for index in range(8):
                trains.append(cluster_start + index * 0.5 + rng.normal(0, 0.03, 20))
                trains.append(cluster_start + index * 0.5 + 0.15 + rng.normal(0, 0.03, 20))
        trains.append(rng.uniform(0, 50, 5))
        spikes[f"u{u}"] = np.sort(np.concatenate(trains))

    return {
        "parameter_free": pf_detect(spikes, duration_s=50.0),
        "gaussian": gauss_detect(spikes, duration_s=50.0),
    }


def _emitted_keys(result):
    """Every metric key the detector produced, as (scope, key) pairs."""
    pairs = set()
    for tier in TIERS:
        section = result.get(tier) or {}
        for event in section.get("events") or []:
            pairs.update(("event", key) for key in event)
        for key in section.get("metrics") or {}:
            pairs.add(("tier_metrics", key))
    for stats in (result.get("unit_stats") or {}).values():
        pairs.update(("unit_stats", key) for key in stats)
    pairs.update(("diagnostics", key) for key in result.get("diagnostics") or {})
    pairs.update(("plot_data", key) for key in result.get("plot_data") or {})
    return pairs


def test_the_dictionary_documents_every_key_both_detectors_emit(dictionary, results):
    entries = dictionary["keys"]
    documented = {(entry.get("scope"), key) for key, entry in entries.items()}
    # A few keys appear in more than one place, e.g. a threshold that is both a
    # diagnostic and the line drawn on the trace.
    documented |= {(scope, key) for key, entry in entries.items()
                   for scope in entry.get("also_in", ())}

    # A tier metric derived from a per-event field is documented by that field.
    event_keys = {key for scope, key in documented if scope == "event"}
    documented |= {("tier_metrics", key) for key in event_keys}

    missing = set()
    for detector, result in results.items():
        for scope, key in _emitted_keys(result):
            if (scope, key) not in documented:
                missing.add(f"{detector}:{scope}.{key}")

    assert not missing, (
        "keys emitted but absent from metrics.json: " + ", ".join(sorted(missing))
    )


def test_every_tier_populated_in_the_fixture(results):
    """Guards the coverage test above: an empty tier would let its keys pass
    unchecked."""
    result = results["parameter_free"]
    for tier in TIERS:
        assert result[tier]["events"], f"{tier} empty; coverage test is vacuous"


def test_the_dictionary_declares_the_same_schema_version_as_the_code(dictionary):
    assert dictionary["schema_version"] == burst_common.SCHEMA_VERSION


def test_units_are_stated_for_every_key(dictionary):
    missing = [key for key, entry in dictionary["keys"].items() if not entry.get("units")]
    assert not missing, f"keys without units: {missing}"


def test_yield_dependent_keys_say_so(dictionary):
    """The two keys that rise with unit count must carry an explicit
    normalised_by of null and a caveat, so nobody compares them across wells
    by accident."""
    for key in ("spikes_per_burst", "burst_peak_hz_array"):
        entry = dictionary["keys"][key]
        assert "normalised_by" in entry and entry["normalised_by"] is None
        assert entry.get("caveat")


def test_keys_whose_meaning_changed_record_the_schema_they_changed_in(dictionary):
    changed = {key for key, entry in dictionary["keys"].items()
               if entry.get("changed_in_schema")}
    assert "burst_peak_hz_per_unit" in changed
    assert "burst_rate_hz" in changed
    for key in changed:
        assert dictionary["keys"][key].get("caveat"), f"{key} changed silently"


def test_the_generated_document_is_up_to_date():
    """docs/metrics.md is rendered from metrics.json; a stale copy is worse
    than none, because it is the one people read."""
    completed = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_metrics_doc.py"), "--check"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
