"""--skip-spikesorting must leave a checkpoint a rerun can resume from."""

import json
import logging
import types

import pytest

from mea_checkpoint import ProcessingStage
from mea_resume import mark_spike_detection_complete, prepare_for_full_run


class _Pipeline:
    """The few members of MEAPipeline these helpers use, writing a real checkpoint file."""

    def __init__(self, tmp_path, stage=ProcessingStage.PREPROCESSING_COMPLETE, **state):
        self.checkpoint_file = tmp_path / "checkpoint.json"
        self.state = {"stage": stage.value, **state}
        self.logger = logging.getLogger("skip-test")
        self.force_restart = False

    def _save_checkpoint(self, stage, **kwargs):
        self.state["stage"] = stage.value
        self.state.update(kwargs)
        self.checkpoint_file.write_text(json.dumps(self.state))

    def should_skip(self):  # same rule as InfraMixin.should_skip
        return self.state["stage"] == ProcessingStage.REPORTS_COMPLETE.value and not self.force_restart


def test_a_finished_detection_only_run_is_checkpointed_as_complete(tmp_path):
    pipeline = _Pipeline(tmp_path)
    mark_spike_detection_complete(pipeline)
    saved = json.loads(pipeline.checkpoint_file.read_text())
    assert saved["stage"] == ProcessingStage.REPORTS_COMPLETE.value
    assert saved["spike_detection_only"] is True


def test_rerunning_skip_spikesorting_does_not_start_over(tmp_path):
    pipeline = _Pipeline(tmp_path)
    mark_spike_detection_complete(pipeline)
    # A second --skip-spikesorting run does not call prepare_for_full_run.
    assert pipeline.should_skip()


def test_a_full_run_after_detection_only_resumes_before_sorting(tmp_path):
    pipeline = _Pipeline(tmp_path)
    mark_spike_detection_complete(pipeline)

    assert prepare_for_full_run(pipeline) is True
    assert pipeline.state["stage"] == ProcessingStage.PREPROCESSING_COMPLETE.value
    assert pipeline.state["spike_detection_only"] is False
    assert not pipeline.should_skip(), "sorting was never done, so the well is not finished"


def test_a_completed_full_run_is_untouched(tmp_path):
    pipeline = _Pipeline(tmp_path, stage=ProcessingStage.REPORTS_COMPLETE)
    assert prepare_for_full_run(pipeline) is False
    assert pipeline.state["stage"] == ProcessingStage.REPORTS_COMPLETE.value
    assert pipeline.should_skip()


def test_a_stale_flag_on_an_unfinished_well_is_cleared(tmp_path):
    pipeline = _Pipeline(tmp_path, stage=ProcessingStage.SORTING_COMPLETE, spike_detection_only=True)
    assert prepare_for_full_run(pipeline) is False
    assert pipeline.state["spike_detection_only"] is False
    assert pipeline.state["stage"] == ProcessingStage.SORTING_COMPLETE.value
