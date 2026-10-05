try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage


def _normalize_resume_from_stage(resume_from):
    if resume_from is None:
        return None

    token = str(resume_from).strip().lower().replace("-", "_")
    if not token:
        return None

    aliases = {
        "preprocess":   "preprocessing",
        "preprocessing": "preprocessing",
        "sorting":      "sorting",
        "sort":         "sorting",
        "merge":        "merge",
        "analyzer":     "analyzer",
        "analyse":      "analyzer",
        "analysis":     "analyzer",
        "report":       "reports",
        "reports":      "reports",
    }
    normalized = aliases.get(token)
    if normalized is None:
        valid = ", ".join(["preprocessing", "sorting", "merge", "analyzer", "reports"])
        raise ValueError(f"Invalid resume_from stage '{resume_from}'. Valid stages: {valid}")
    return normalized


def _apply_resume_from_stage(pipeline, resume_from):
    stage_name = _normalize_resume_from_stage(resume_from)
    if stage_name is None:
        return

    resume_checkpoint_stage = {
        "preprocessing": ProcessingStage.NOT_STARTED,
        "sorting":       ProcessingStage.PREPROCESSING_COMPLETE,
        "merge":         ProcessingStage.SORTING_COMPLETE,
        "analyzer":      ProcessingStage.MERGE_COMPLETE,
        "reports":       ProcessingStage.ANALYZER_COMPLETE,
    }[stage_name]

    if stage_name in {"merge", "analyzer", "reports"}:
        pipeline.force_rerun_analyzer = True

    pipeline._save_checkpoint(
        resume_checkpoint_stage,
        failed_stage=None,
        error=None,
        resume_from=stage_name,
        resume_forced_rerun_analyzer=bool(pipeline.force_rerun_analyzer),
    )
    pipeline.logger.info(
        "Resume-from requested: %s (checkpoint set to %s)",
        stage_name,
        resume_checkpoint_stage.name,
    )


# ---------------------------------------------------------------------------
# --skip-spikesorting checkpointing
#
# The detection-only path never sorts, so it used to leave the checkpoint at
# PREPROCESSING_COMPLETE: a rerun redid everything, and the checkpoint showed a
# finished well as unfinished. It now records REPORTS_COMPLETE plus a
# spike_detection_only flag, so a rerun skips, and a later full run knows the
# sorting stages were never done.
# ---------------------------------------------------------------------------

def mark_spike_detection_complete(pipeline):
    """Checkpoint a finished --skip-spikesorting run."""
    pipeline._save_checkpoint(
        ProcessingStage.REPORTS_COMPLETE,
        failed_stage=None,
        error=None,
        spike_detection_only=True,
        note="Spike detection + burst analysis (no sorting)",
    )


def prepare_for_full_run(pipeline):
    """Call before a run that will sort. A well completed by detection-only has
    stage REPORTS_COMPLETE but no sorter output, so rewind it to
    PREPROCESSING_COMPLETE (the binary cache is reused) and clear the flag.
    Returns True when it rewound."""
    state = pipeline.state
    if not state.get("spike_detection_only"):
        return False
    if int(state.get("stage", 0)) == ProcessingStage.REPORTS_COMPLETE.value:
        pipeline._save_checkpoint(
            ProcessingStage.PREPROCESSING_COMPLETE,
            spike_detection_only=False,
            failed_stage=None,
            error=None,
        )
        pipeline.logger.info(
            "Previous run was detection-only (no sorting); resuming from PREPROCESSING_COMPLETE."
        )
        return True
    state["spike_detection_only"] = False
    return False
