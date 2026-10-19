"""binary/ reuse: the completion marker and the --stop-after option."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mea_preprocessing as mp
from mea_analysis_routine import MEARunOptions, run_mea_pipeline


def _fake_binary(folder, nbytes=1024):
    folder.mkdir(parents=True)
    (folder / "traces_cached_seg0.raw").write_bytes(b"\0" * nbytes)
    return folder


def test_a_missing_folder_is_missing(tmp_path):
    assert mp.binary_status(tmp_path / "binary") == "missing"


def test_a_folder_without_a_marker_is_unmarked(tmp_path):
    folder = _fake_binary(tmp_path / "binary")
    assert mp.binary_status(folder) == "unmarked"


def test_a_marked_folder_is_complete(tmp_path):
    folder = _fake_binary(tmp_path / "binary")
    mp.write_binary_sentinel(folder, n_channels=4, n_frames=256)
    assert mp.binary_status(folder) == "complete"


def test_a_marker_that_disagrees_with_the_files_is_invalid(tmp_path):
    folder = _fake_binary(tmp_path / "binary")
    mp.write_binary_sentinel(folder)
    (folder / "traces_cached_seg0.raw").write_bytes(b"\0" * 10)  # truncated afterwards
    assert mp.binary_status(folder) == "invalid"


def test_a_marker_from_another_preprocessing_version_is_invalid(tmp_path):
    folder = _fake_binary(tmp_path / "binary")
    mp.write_binary_sentinel(folder)
    path = folder / mp.BINARY_SENTINEL
    info = json.loads(path.read_text())
    info["preprocessing_version"] = mp.PREPROCESSING_VERSION + 1
    path.write_text(json.dumps(info))
    assert mp.binary_status(folder) == "invalid"


def test_an_empty_binary_is_never_complete(tmp_path):
    folder = _fake_binary(tmp_path / "binary", nbytes=0)
    mp.write_binary_sentinel(folder)
    assert mp.binary_status(folder) == "invalid"


@pytest.mark.parametrize("bad", ["preprocess", "analyzer", ""])
def test_stop_after_rejects_unknown_stages(bad):
    with pytest.raises(ValueError):
        run_mea_pipeline(MEARunOptions(file_path="x.h5", stream_id="well000", stop_after=bad))


def test_stop_after_cannot_be_combined_with_detection_only():
    with pytest.raises(ValueError):
        run_mea_pipeline(MEARunOptions(file_path="x.h5", stream_id="well000",
                                       stop_after="preprocessing", skip_spikesorting=True))
