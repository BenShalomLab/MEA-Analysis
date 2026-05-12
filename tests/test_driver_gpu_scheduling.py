import logging
import sys
import types
import importlib

import pytest


def _load_driver_module():
    # Allow importing run_pipeline_driver even when heavy deps are unavailable.
    sys.modules.setdefault("h5py", types.SimpleNamespace(File=None))
    sys.modules.setdefault("pandas", types.SimpleNamespace(read_excel=None))
    return importlib.import_module("run_pipeline_driver")


def _logger():
    logger = logging.getLogger("test_driver_gpu_scheduling")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    return logger


def test_parse_gpu_ids_basic_and_dedup():
    driver = _load_driver_module()
    assert driver._parse_gpu_ids("0,1,2,2") == [0, 1, 2]
    assert driver._parse_gpu_ids("  ") is None
    assert driver._parse_gpu_ids(None) is None


def test_parse_gpu_ids_invalid_raises():
    driver = _load_driver_module()
    with pytest.raises(ValueError):
        driver._parse_gpu_ids("gpu0,1")


def test_determine_max_concurrency_skip_sorting_default():
    driver = _load_driver_module()
    effective, gpu_ids = driver._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": True, "max_concurrent_wells": None, "gpu_ids": None},
        logger=_logger(),
    )
    assert effective == driver.DEFAULT_SKIP_SORTING_CONCURRENCY
    assert gpu_ids is None


def test_determine_max_concurrency_multi_gpu_defaults_to_gpu_count():
    driver = _load_driver_module()
    effective, gpu_ids = driver._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": False, "max_concurrent_wells": None, "gpu_ids": "0,1,2,3"},
        logger=_logger(),
    )
    assert effective == 4
    assert gpu_ids == [0, 1, 2, 3]


def test_determine_max_concurrency_multi_gpu_respects_cap():
    driver = _load_driver_module()
    effective, gpu_ids = driver._determine_max_concurrency(
        args=types.SimpleNamespace(),
        resolved={"skip_spikesorting": False, "max_concurrent_wells": 2, "gpu_ids": "0,1,2,3"},
        logger=_logger(),
    )
    assert effective == 2
    assert gpu_ids == [0, 1, 2, 3]

