"""--burst-detector: one detector or "both", with outputs that never collide."""

import logging
import sys
import types

import numpy as np
import pytest

import config_loader as cfg


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_both_is_the_default():
    resolved = cfg.resolve_args(types.SimpleNamespace(), {})
    assert resolved["burst_detector"] == "both"


def test_cli_value_overrides_the_config_file():
    args = types.SimpleNamespace(burst_detector="gaussian")
    config = {"burst_detection": {"burst_detector": "parameter_free"}}
    assert cfg.resolve_args(args, config)["burst_detector"] == "gaussian"


def test_both_runs_every_detector_in_a_fixed_order():
    assert cfg.resolve_burst_detectors("both") == ["parameter_free", "gaussian"]
    assert cfg.resolve_burst_detectors(None) == ["parameter_free", "gaussian"]
    assert cfg.resolve_burst_detectors("Gaussian") == ["gaussian"]
    assert cfg.resolve_burst_detectors("parameter_free") == ["parameter_free"]


def test_unknown_detector_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="both"):
        cfg.resolve_burst_detectors("nope")


def test_a_bad_value_in_the_config_file_fails_early():
    config = {"burst_detection": {"burst_detector": "nope"}}
    with pytest.raises(ValueError):
        cfg.resolve_args(types.SimpleNamespace(), config)


def test_the_primary_keeps_canonical_names_and_the_other_is_suffixed():
    assert cfg.burst_file_suffix("parameter_free", "both") == ""
    assert cfg.burst_file_suffix("gaussian", "both") == "_gaussian"


def test_a_single_detector_always_uses_canonical_names():
    assert cfg.burst_file_suffix("gaussian", "gaussian") == ""
    assert cfg.burst_file_suffix("parameter_free", "parameter_free") == ""


def test_the_driver_forwards_the_selection_to_each_well():
    resolved = cfg.resolve_args(types.SimpleNamespace(burst_detector="both"), {})
    extra = cfg.build_extra_args(resolved, types.SimpleNamespace(config=None))
    assert "--burst-detector both" in extra


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------

def test_the_collector_reads_both_detectors_and_ignores_temp_files(tmp_path):
    import json
    from pathlib import Path
    import collect_network_jsons as collector

    well = tmp_path / "proj" / "240531" / "chip" / "run_001" / "Network" / "well000"
    well.mkdir(parents=True)
    base = {"n_units": 12, "network_bursts": {"events": [], "metrics": {"burst_count": 0}},
            "project": "proj", "run_id": "run_001", "well": "well000"}
    (well / "network_results.json").write_text(json.dumps({**base, "detector": "parameter_free"}))
    (well / "network_results_gaussian.json").write_text(json.dumps({**base, "detector": "gaussian"}))
    (well / "network_results.tmp.json").write_text("{ interrupted")

    rows = collector.collect(Path(tmp_path))
    assert sorted(r["detector"] for r in rows) == ["gaussian", "parameter_free"]
    assert {r["result_file"] for r in rows} == {"network_results.json", "network_results_gaussian.json"}


# ---------------------------------------------------------------------------
# report generation
# ---------------------------------------------------------------------------

def _spikes(n_units=25, duration=120.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for u in range(n_units):
        trains = [c + rng.normal(0, 0.05, 40) for c in np.arange(10, duration - 10, 10.0)]
        trains.append(rng.uniform(0, duration, 30))
        out[u] = np.sort(np.concatenate(trains))
    return out


class _FakeRecording:
    def get_sampling_frequency(self):
        return 10000.0

    def get_num_frames(self):
        return int(120.0 * 10000.0)


@pytest.fixture
def reports():
    pytest.importorskip("matplotlib.pyplot")
    try:
        import spikeinterface.full  # noqa: F401
    except ImportError:
        for name in ("spikeinterface", "spikeinterface.full"):
            sys.modules.setdefault(name, types.ModuleType(name))
    import mea_reports
    return mea_reports


def _run(tmp_path, reports, selection):
    np.save(tmp_path / "spike_times.npy", _spikes())

    class Harness(reports.ReportsMixin):
        pass

    h = Harness()
    h.output_dir = tmp_path
    h.output_root = tmp_path / "root"
    h.logger = logging.getLogger("selection-test")
    h.sorting = None
    h.recording = _FakeRecording()
    h.burst_detector = selection
    h.gaussian_burst_kwargs = {}
    h.parameter_free_burst_kwargs = {}
    h.curation_summary = None
    h.project_name, h.date, h.chip_id, h.run_id, h.well = "P", "d", "c", "r", "w"
    h._run_burst_analysis()
    return h


def _json(path):
    import json
    return json.loads(path.read_text())


PRIMARY_FILES = ["network_results.json", "network_plot_data.npz", "raster_burst_plot.svg",
                 "raster_burst_plot_60s.svg", "raster_burst_plot_30s.svg"]
SECONDARY_FILES = ["network_results_gaussian.json", "network_plot_data_gaussian.npz",
                   "raster_burst_plot_gaussian.svg", "raster_burst_plot_gaussian_60s.svg",
                   "raster_burst_plot_gaussian_30s.svg"]


def test_both_writes_every_detectors_files_without_collisions(tmp_path, reports):
    _run(tmp_path, reports, "both")
    for name in PRIMARY_FILES + SECONDARY_FILES:
        assert (tmp_path / name).exists(), name
    assert _json(tmp_path / "network_results.json")["detector"] == "parameter_free"
    assert _json(tmp_path / "network_results_gaussian.json")["detector"] == "gaussian"
    assert not list(tmp_path.glob("*.tmp.json"))


def test_one_detector_writes_only_its_own_canonical_files(tmp_path, reports):
    _run(tmp_path, reports, "parameter_free")
    for name in PRIMARY_FILES:
        assert (tmp_path / name).exists(), name
    assert not [p for p in tmp_path.iterdir() if "gaussian" in p.name]


def test_gaussian_alone_takes_the_canonical_names(tmp_path, reports):
    _run(tmp_path, reports, "gaussian")
    assert _json(tmp_path / "network_results.json")["detector"] == "gaussian"
    assert not [p for p in tmp_path.iterdir() if "gaussian" in p.name]


def test_one_detector_failing_does_not_lose_the_other(tmp_path, reports, monkeypatch):
    def boom(**kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setitem(reports.BURST_DETECTORS, "gaussian", boom)
    with pytest.raises(RuntimeError, match="gaussian"):
        _run(tmp_path, reports, "both")
    assert _json(tmp_path / "network_results.json")["detector"] == "parameter_free"
    assert not (tmp_path / "network_results_gaussian.json").exists()
