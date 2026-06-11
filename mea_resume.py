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
