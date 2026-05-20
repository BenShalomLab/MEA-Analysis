import logging
import sys
import types
import importlib
import uuid

import pytest


@pytest.fixture
def driver_module(monkeypatch):
    # Allow importing run_pipeline_driver even when heavy deps are unavailable.
    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=None))
    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace(read_excel=None))
    sys.modules.pop("run_pipeline_driver", None)
    module = importlib.import_module("run_pipeline_driver")
    yield module
    sys.modules.pop("run_pipeline_driver", None)


def _logger():
    logger = logging.getLogger(f"test_driver_gpu_scheduling_{uuid.uuid4().hex}")
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def test_parse_gpu_ids_basic_and_dedup(driver_module):
    assert driver_module._parse_gpu_ids("0,1,2,2") == [0, 1, 2]
    assert driver_module._parse_gpu_ids("  ") is None
    assert driver_module._parse_gpu_ids(None) is None


def test_parse_gpu_ids_invalid_raises(driver_module):
    with pytest.raises(ValueError):
        driver_module._parse_gpu_ids("gpu0,1")


def test_determine_max_concurrency_skip_sorting_default(driver_module):
    effective, gpu_ids = driver_module._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": True, "max_concurrent_wells": None, "gpu_ids": None},
        logger=_logger(),
    )
    assert effective == driver_module.DEFAULT_SKIP_SORTING_CONCURRENCY
    assert gpu_ids is None


def test_determine_max_concurrency_multi_gpu_defaults_to_gpu_count(driver_module):
    effective, gpu_ids = driver_module._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": False, "max_concurrent_wells": None, "gpu_ids": "0,1,2,3"},
        logger=_logger(),
    )
    assert effective == 4
    assert gpu_ids == [0, 1, 2, 3]


def test_determine_max_concurrency_multi_gpu_respects_cap(driver_module):
    effective, gpu_ids = driver_module._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": False, "max_concurrent_wells": 2, "gpu_ids": "0,1,2,3"},
        logger=_logger(),
    )
    assert effective == 2
    assert gpu_ids == [0, 1, 2, 3]
