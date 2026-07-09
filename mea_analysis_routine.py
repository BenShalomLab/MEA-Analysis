# ==========================================================
# mea_analysis_routine.py
# Author: Mandar Patil
# Contributors: Yuxin Ren, Shruti Shah, Adam Weiner
# LLM Assisted Edits: Yes ChatGPT-4, Claude sonnet 4.6, ChatGPT-5.3-Codex
# ==========================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # For headless environments
import sys
import shutil
import gc
import json
import re
import traceback
import logging
import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from enum import Enum
from math import floor
from typing import Any
import psutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# Scientific Libraries
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import spikeinterface.curation as sic

# If this file is executed as a *script* (not imported as a package module),
# allow local imports by adding the script folder and repo root to sys.path.
if not __package__:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))

    root_dir = script_dir.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))


try:
    if __package__:
        from .config_loader import load_config, resolve_args
        from .mea_checkpoint import ProcessingStage, CHECKPOINT_SCHEMA_VERSION
        from .mea_infra import InfraMixin
        from .mea_preprocessing import PreprocessingMixin
        from .mea_sorting import SortingMixin
        from .mea_merge import MergeMixin
        from .mea_analyzer import AnalyzerMixin
        from .mea_waveform import WaveformMixin
        from .mea_reports import ReportsMixin
        from .mea_resume import _normalize_resume_from_stage, _apply_resume_from_stage
    else:
        from config_loader import load_config, resolve_args
        from mea_checkpoint import ProcessingStage, CHECKPOINT_SCHEMA_VERSION
        from mea_infra import InfraMixin
        from mea_preprocessing import PreprocessingMixin
        from mea_sorting import SortingMixin
        from mea_merge import MergeMixin
        from mea_analyzer import AnalyzerMixin
        from mea_waveform import WaveformMixin
        from mea_reports import ReportsMixin
        from mea_resume import _normalize_resume_from_stage, _apply_resume_from_stage
except ImportError:
    try:
        from MEA_Analysis.IPNAnalysis.config_loader import load_config, resolve_args
        from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage, CHECKPOINT_SCHEMA_VERSION
        from MEA_Analysis.IPNAnalysis.mea_infra import InfraMixin
        from MEA_Analysis.IPNAnalysis.mea_preprocessing import PreprocessingMixin
        from MEA_Analysis.IPNAnalysis.mea_sorting import SortingMixin
        from MEA_Analysis.IPNAnalysis.mea_merge import MergeMixin
        from MEA_Analysis.IPNAnalysis.mea_analyzer import AnalyzerMixin
        from MEA_Analysis.IPNAnalysis.mea_waveform import WaveformMixin
        from MEA_Analysis.IPNAnalysis.mea_reports import ReportsMixin
        from MEA_Analysis.IPNAnalysis.mea_resume import _normalize_resume_from_stage, _apply_resume_from_stage
    except ImportError as e:
        raise ImportError(
            "Could not import MEA_Analysis.IPNAnalysis helper modules. "
            "If you are running this file directly, prefer: "
            "`python -m MEA_Analysis.IPNAnalysis.mea_analysis_routine ...`"
        ) from e


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


# --- Default kwarg factories ---
def _default_um_kwargs() -> dict[str, Any]:
    return {
        "merge_units": False,
        "dry_run": True,
        "scored_dry_run": True,
        "output_subdir_name": "unitmatch_outputs",
        "throughput_subdir_name": "unitmatch_throughput",
        "max_candidate_pairs": 20000,
        "oversplit_min_probability": 0.99,
        "oversplit_max_suggestions": 2000,
        "apply_merges": False,
        "recursive": False,
        "max_iterations": 5,
        "max_spikes_per_unit": 100,
        "keep_all_iterations": True,
        "generate_reports": True,
        "report_subdir_name": "unitmatch_reports",
        "report_max_heatmap_units": 200,
    }


def _default_am_kwargs() -> dict[str, Any]:
    return {
        "enabled": False,
        "presets": None,
        "steps_params": None,
        "template_diff_thresh": "0.05,0.15,0.25",
    }


def _default_option_kwargs() -> dict[str, Any]:
    return {
        "force_rerun_analyzer": False,
        "preprocessed_recording": None,
        "skip_preprocessing": False,
        "cuda_visible_devices": None,
        "output_subdir_after_well": None,
    }


# --- The Main Pipeline Class ---
class MEAPipeline(
    InfraMixin,
    PreprocessingMixin,
    SortingMixin,
    MergeMixin,
    AnalyzerMixin,
    WaveformMixin,
    ReportsMixin,
):
    """
    SOTA Pipeline for 2D MEA Analysis (Maxwell Biosystems).
    Encapsulates Preprocessing, Sorting, Analysis, and Curation.
    """

    def __init__(self, file_path, stream_id='well000', recording_num='rec0000', output_root=None,
                 checkpoint_root=None, sorter='kilosort4', docker_image=None, verbose=True,
                 cleanup=False, force_restart=False,
                 n_jobs: int | None = None,
                 chunk_duration: str | None = None,
                 sorter_kwargs: dict | None = None,
                 um_kwargs: dict | None = None,
                 am_kwargs: dict | None = None,
                 option_kwargs: dict | None = None):

        self.file_path = Path(file_path).resolve()
        self.stream_id = stream_id
        self.recording_num = recording_num
        self.sorter = sorter
        self.docker_image = docker_image
        self.verbose = verbose
        self.cleanup_flag = cleanup
        self.force_restart = force_restart

        self.um_kwargs = _default_um_kwargs()
        if isinstance(um_kwargs, dict):
            self.um_kwargs.update(um_kwargs)
        self.am_kwargs = _default_am_kwargs()
        if isinstance(am_kwargs, dict):
            self.am_kwargs.update(am_kwargs)
        self.option_kwargs = _default_option_kwargs()
        if isinstance(option_kwargs, dict):
            self.option_kwargs.update(option_kwargs)

        self.n_jobs = n_jobs
        self.chunk_duration = chunk_duration
        self.output_subdir_after_well = self._validate_output_subdir_after_well(self.option_kwargs.get("output_subdir_after_well"))

        self.sorter_kwargs = sorter_kwargs

        self.unitmatch_merge_units = bool(self.um_kwargs.get("merge_units"))
        self.unitmatch_dry_run = bool(self.um_kwargs.get("dry_run"))
        self.unitmatch_scored_dry_run = bool(self.um_kwargs.get("scored_dry_run"))
        self.unitmatch_output_subdir_name = (
            self._validate_output_subdir_after_well(self.um_kwargs.get("output_subdir_name"))
            or "unitmatch_outputs"
        )
        self.unitmatch_throughput_subdir_name = (
            self._validate_output_subdir_after_well(self.um_kwargs.get("throughput_subdir_name"))
            or "unitmatch_throughput"
        )
        self.unitmatch_max_candidate_pairs = int(self.um_kwargs.get("max_candidate_pairs"))
        self.unitmatch_oversplit_min_probability = float(self.um_kwargs.get("oversplit_min_probability"))
        self.unitmatch_oversplit_max_suggestions = int(self.um_kwargs.get("oversplit_max_suggestions"))
        self.unitmatch_apply_merges = bool(self.um_kwargs.get("apply_merges"))
        self.unitmatch_recursive = bool(self.um_kwargs.get("recursive"))
        self.unitmatch_max_iterations = int(self.um_kwargs.get("max_iterations"))
        self.unitmatch_max_spikes_per_unit = int(self.um_kwargs.get("max_spikes_per_unit"))
        self.unitmatch_keep_all_iterations = bool(self.um_kwargs.get("keep_all_iterations"))
        self.unitmatch_generate_reports = bool(self.um_kwargs.get("generate_reports"))
        self.unitmatch_report_subdir_name = (
            self._validate_output_subdir_after_well(self.um_kwargs.get("report_subdir_name"))
            or "unitmatch_reports"
        )
        self.unitmatch_report_max_heatmap_units = int(self.um_kwargs.get("report_max_heatmap_units"))

        self.auto_merge_units = bool(self.am_kwargs.get("enabled"))
        self.auto_merge_presets = self.am_kwargs.get("presets")
        self.auto_merge_steps_params = self.am_kwargs.get("steps_params")

        self.force_rerun_analyzer = bool(self.option_kwargs.get("force_rerun_analyzer"))
        self.preprocessed_recording = self.option_kwargs.get("preprocessed_recording")
        self.skip_preprocessing = bool(self.option_kwargs.get("skip_preprocessing"))
        self.cuda_visible_devices = self.option_kwargs.get("cuda_visible_devices")

        # 1. Parse Metadata & Paths
        self.metadata = self._parse_metadata()
        self.run_id = self.metadata.get('run_id', 'UnknownRun')
        self.project_name = self.metadata.get('project', 'UnknownProject')
        self.well = self.metadata.get('well', 'UnknownWell')
        self.chip_id = self.metadata.get('chip_id', 'UnknownChip')
        self.date = self.metadata.get('date', 'UnknownDate')

        self.relative_pattern = self.metadata.get('relative_pattern', 'UnknownPattern')
        default_output_root = Path(__file__).resolve().parent / "AnalyzedData"
        effective_output_root = Path(output_root) if output_root is not None else default_output_root
        self.output_root = effective_output_root
        base_output_dir = effective_output_root / self.relative_pattern / self.stream_id
        self.output_dir = (
            base_output_dir / self.output_subdir_after_well
            if self.output_subdir_after_well is not None
            else base_output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Logger Setup
        log_file = self.output_dir / f"{self.run_id}_{self.stream_id}_pipeline.log"
        self.logger = self._setup_logger(log_file)

        # 3. Checkpointing Setup
        # Primary checkpoint always lives inside the well output folder.
        # If checkpoint_root is also provided it receives an additional mirrored copy.
        ckpt_fname = f"{self.project_name}_{self.run_id}_{self.stream_id}_checkpoint.json"
        default_ckpt_root = self.output_dir / "checkpoints"
        default_ckpt_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = default_ckpt_root / ckpt_fname

        self.extra_checkpoint_file = None
        if checkpoint_root is not None:
            extra_root = Path(checkpoint_root)
            if extra_root.resolve() != default_ckpt_root.resolve():
                extra_root.mkdir(parents=True, exist_ok=True)
                self.extra_checkpoint_file = extra_root / ckpt_fname

        self.state = self._load_checkpoint()

        self._apply_runtime_controls()
        self._log_runtime_controls()

        # Data placeholders
        self.recording = None
        self.sorting = None
        self.analyzer = None

    def cleanup(self):
        if self.cleanup_flag:
            self.logger.info("Cleaning up temp files...")
            # Preserve binary/ when phy_output exists: phy's symlink relies on it.
            if (self.output_dir / "phy_output").exists():
                self.logger.info("Skipping binary/ cleanup: phy_output exists and symlinks into it.")
            else:
                shutil.rmtree(self.output_dir / "binary", ignore_errors=True)
            shutil.rmtree(self.output_dir / "sorter_output", ignore_errors=True)
        self.recording = None
        self.sorting = None
        self.analyzer = None
        gc.collect()


@dataclass(frozen=True)
class MEARunOptions:
    file_path: str | Path
    stream_id: str
    recording_num: str = "rec0000"
    output_root: str | Path | None = None
    checkpoint_root: str | Path | None = None
    output_subdir_after_well: str | None = None
    sorter: str = "kilosort4"
    docker_image: str | None = None
    verbose: bool = False
    cleanup: bool = False
    force_restart: bool = False
    resume_from: str | None = None
    n_jobs: int | None = None
    chunk_duration: str | None = None
    sorter_kwargs: dict | None = None
    um_kwargs: dict | None = None
    am_kwargs: dict | None = None
    option_kwargs: dict | None = None
    reanalyze_bursts: bool = False
    skip_spikesorting: bool = False
    run_analyzer: bool = True
    run_reports: bool = True
    thresholds: dict | None = None
    no_curation: bool = False
    export_to_phy: bool = False
    plot_mode: str = "separate"
    plot_debug: bool = False
    raster_sort: str | None = None
    fixed_y: bool = False
    auto_merge_template_diff_thresh: str = "0.05,0.15,0.25"
    extract_rawsortedspikes: bool = False


@dataclass(frozen=True)
class MEARunResult:
    pipeline: MEAPipeline
    skipped: bool = False
    reanalyzed_bursts: bool = False


def run_mea_pipeline(options: MEARunOptions) -> MEARunResult:
    um_kwargs = _default_um_kwargs()
    if isinstance(options.um_kwargs, dict):
        um_kwargs.update(options.um_kwargs)

    am_kwargs = _default_am_kwargs()
    if isinstance(options.am_kwargs, dict):
        am_kwargs.update(options.am_kwargs)

    option_kwargs = _default_option_kwargs()
    if isinstance(options.option_kwargs, dict):
        option_kwargs.update(options.option_kwargs)

    auto_merge_presets = am_kwargs.get("presets")
    auto_merge_steps_params = am_kwargs.get("steps_params")

    if bool(am_kwargs.get("enabled")) and (auto_merge_presets is None or auto_merge_steps_params is None):
        try:
            diffs = [
                float(x.strip())
                for x in str(am_kwargs.get("template_diff_thresh", "0.05,0.15,0.25")).split(",")
                if x.strip()
            ]
            auto_merge_presets = ["x_contaminations"] * len(diffs)
            auto_merge_steps_params = [
                {"template_similarity": {"template_diff_thresh": float(t)}}
                for t in diffs
            ]
        except Exception:
            auto_merge_presets = None
            auto_merge_steps_params = None

    am_kwargs["presets"] = auto_merge_presets
    am_kwargs["steps_params"] = auto_merge_steps_params

    pipeline = MEAPipeline(
        file_path=options.file_path,
        stream_id=options.stream_id,
        recording_num=options.recording_num,
        output_root=options.output_root,
        checkpoint_root=options.checkpoint_root,
        sorter=options.sorter,
        docker_image=options.docker_image,
        verbose=bool(options.verbose),
        cleanup=bool(options.cleanup),
        force_restart=bool(options.force_restart),
        n_jobs=options.n_jobs,
        chunk_duration=options.chunk_duration,
        sorter_kwargs=options.sorter_kwargs,
        um_kwargs=um_kwargs,
        am_kwargs=am_kwargs,
        option_kwargs=option_kwargs,
    )

    # skip_spikesorting still executes spike-detection + burst analysis, so it counts as execution.
    stage_execution_requested = any(
        (
            options.reanalyze_bursts,
            options.skip_spikesorting,
            options.run_analyzer,
            options.run_reports,
        )
    )
    cleanup_only = options.cleanup and not stage_execution_requested
    if cleanup_only:
        pipeline.logger.info("Running in cleanup-only mode; no processing stages will execute.")
        pipeline.cleanup()
        return MEARunResult(pipeline=pipeline, skipped=False, reanalyzed_bursts=False)

    _apply_resume_from_stage(pipeline, options.resume_from)
    uses_spike_sorting = not bool(options.skip_spikesorting)
    pipeline._write_processing_info(
        used_spike_sorting=uses_spike_sorting,
        reanalyze_bursts=bool(options.reanalyze_bursts),
        extract_rawsortedspikes=bool(options.extract_rawsortedspikes),
    )

    if bool(options.reanalyze_bursts):
        if bool(options.extract_rawsortedspikes):
            pipeline._extract_rawsortedspikes()
        pipeline._run_burst_analysis(
            plot_mode=options.plot_mode,
            plot_debug=bool(options.plot_debug),
            raster_sort=options.raster_sort,
            fixed_y=bool(options.fixed_y),
        )
        pipeline._save_checkpoint(
            ProcessingStage.REPORTS_COMPLETE,
            note="Burst Re-analysis Performed",
            last_updated=str(datetime.now()),
        )
        return MEARunResult(pipeline=pipeline, skipped=False, reanalyzed_bursts=True)

    if pipeline.should_skip():
        return MEARunResult(pipeline=pipeline, skipped=True, reanalyzed_bursts=False)

    pipeline.run_preprocessing()

    if not bool(options.skip_spikesorting):
        pipeline.run_sorting()
        pipeline.run_optional_merge_phase()

        if bool(options.run_analyzer):
            pipeline.run_analyzer()
            if bool(options.extract_rawsortedspikes):
                pipeline._extract_rawsortedspikes()

        if bool(options.run_reports):
            if (not bool(options.run_analyzer)) and pipeline.analyzer is None:
                raise RuntimeError(
                    "Requested report generation but analyzer was not run and no existing analyzer was loaded. "
                    "Set run_analyzer=True or run once to populate analyzer_output."
                )
            pipeline.generate_reports(
                options.thresholds,
                bool(options.no_curation),
                bool(options.export_to_phy),
                plot_mode=options.plot_mode,
                plot_debug=bool(options.plot_debug),
                raster_sort=options.raster_sort,
                fixed_y=bool(options.fixed_y),
            )
    else:
        if bool(options.extract_rawsortedspikes):
            pipeline.logger.warning(
                "--extract-rawsortedspikes requested but --skip-spikesorting is enabled; skipping extraction."
            )
        ids = pipeline._spike_detection_only()
        pipeline._run_burst_analysis(
            ids,
            plot_mode=options.plot_mode,
            plot_debug=bool(options.plot_debug),
            raster_sort=options.raster_sort,
            fixed_y=bool(options.fixed_y),
        )

    if bool(options.cleanup):
        pipeline.cleanup()

    return MEARunResult(pipeline=pipeline, skipped=False, reanalyzed_bursts=False)


# --- CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser(
        description="MEA Analysis Routine — processes a single well from an MEA recording",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Positional ---
    parser.add_argument("file_path",
        help="Path to .h5, .nwb, or .raw MEA recording file")

    # --- Input / Output ---
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--config", type=str, default=None,
        help="Path to config JSON file (CLI flags always override config)")
    io_group.add_argument("--well", required=True,
        help="Well ID to process (e.g. well000)")
    io_group.add_argument("--rec", type=str, default=None,
        help="Recording name inside HDF5 file (default: rec0000)")
    io_group.add_argument("--output-dir", type=str, default=None,
        help="Output directory for results (default: <repo>/AnalyzedData)")
    io_group.add_argument("--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (default: <output-dir>/checkpoints)")
    io_group.add_argument("--output-subdir-after-well", type=str, default=None,
        help="Optional single subdirectory appended under the resolved well output directory")
    io_group.add_argument("--export-to-phy", action="store_true",
        help="Export results to Phy format")
    io_group.add_argument("--clean-up", action="store_true",
        help="Remove intermediate files after processing")

    # --- Sorting ---
    sort_group = parser.add_argument_group("sorting")
    sort_group.add_argument("--sorter", type=str, default=None,
        help="Spike sorter to use (default: kilosort4)")
    sort_group.add_argument("--docker", type=str, default=None,
        help="Docker image name for containerized sorting")
    sort_group.add_argument("--skip-spikesorting", action="store_true",
        help="Run spike detection only, skip full sorting")

    # --- Plotting ---
    plot_group = parser.add_argument_group("plotting")
    plot_group.add_argument("--plot-mode", choices=["separate", "merged"], default=None,
        help="Plot raster and network on separate axes or merged twin-axis\n(default: separate)")
    plot_group.add_argument("--raster-sort", choices=["none", "firing_rate", "location_y", "unit_id"], default=None,
        help="How to sort units on raster y-axis (default: none)")
    plot_group.add_argument("--plot-debug", action="store_true",
        help="Overlay burst and superburst intervals on raster plot")
    plot_group.add_argument("--fixed-y", action="store_true",
        help="Use fixed y-axis limits for raster plots — run once without it first to generate summary")

    # --- Curation ---
    cur_group = parser.add_argument_group("curation")
    cur_group.add_argument("--no-curation", action="store_true",
        help="Skip automatic unit curation")
    cur_group.add_argument("--params", type=str, default=None,
        help="JSON string or file path with quality thresholds")

    # --- Run Control ---
    ctrl_group = parser.add_argument_group("run control")
    ctrl_group.add_argument("--force-restart", action="store_true",
        help="Ignore checkpoint and restart from scratch")
    ctrl_group.add_argument("--resume-from", "--resume_from", dest="resume_from", type=str, default=None,
        choices=["preprocessing", "sorting", "merge", "analyzer", "reports"],
        help="Resume by rewinding checkpoint to just before this stage and rerunning from there")
    ctrl_group.add_argument("--reanalyze-bursts", action="store_true",
        help="Re-run burst analysis on existing spike times only")
    ctrl_group.add_argument("--extract-rawsortedspikes", action="store_true",
        help="Extract per-unit raw mean templates and save raw_mean_templates.npy (requires analyzer_output or phy_output)")
    ctrl_group.add_argument("--debug", action="store_true",
        help="Enable verbose logging")

    # Optional post-spikesort step(s)
    parser.add_argument("--unitmatch-merge-units", action='store_true',
        help="Optional (default off): run UnitMatch integration as an alternative to auto_merge_units.")
    parser.add_argument("--unitmatch-dry-run", action='store_true',
        help="When UnitMatch is enabled, produce UnitMatch reports without applying merges.")
    parser.add_argument("--unitmatch-scored-dry-run", action=argparse.BooleanOptionalAction, default=None,
        help="When UnitMatch dry-run is enabled, attempt backend scoring (default: enabled).")
    parser.add_argument("--unitmatch-output-subdir-name", type=str, default=None,
        help="UnitMatch artifact subdirectory under output_dir (default: unitmatch_outputs).")
    parser.add_argument("--unitmatch-throughput-subdir-name", type=str, default=None,
        help="UnitMatch throughput subdirectory under output_dir (default: unitmatch_throughput).")
    parser.add_argument("--unitmatch-max-candidate-pairs", type=int, default=None,
        help="Maximum UnitMatch candidate pairs (-1 unlimited, 0 none, default: 20000).")
    parser.add_argument("--unitmatch-oversplit-min-probability", type=float, default=None,
        help="Minimum UnitMatch probability for oversplit suggestions (default: 0.80).")
    parser.add_argument("--unitmatch-oversplit-max-suggestions", type=int, default=None,
        help="Maximum oversplit suggestions (-1 unlimited, 0 none, default: 2000).")
    parser.add_argument("--unitmatch-apply-merges", action='store_true',
        help="Apply conflict-free top UnitMatch suggestions to mutate sorting.")
    parser.add_argument("--unitmatch-recursive", action='store_true',
        help="Recursively run UnitMatch merge iterations until convergence or cap.")
    parser.add_argument("--unitmatch-max-iterations", type=int, default=None,
        help="Maximum recursive UnitMatch iterations (-1 uncapped, default: 5).")
    parser.add_argument("--unitmatch-max-spikes-per-unit", type=int, default=None,
        help="Max spikes per unit for UnitMatch raw-waveform generation (-1 uncapped, default: 100).")
    parser.add_argument("--unitmatch-keep-all-iterations", action=argparse.BooleanOptionalAction, default=None,
        help="Keep all unitmatch_throughput iteration folders (default: enabled).")
    parser.add_argument("--unitmatch-generate-reports", action=argparse.BooleanOptionalAction, default=True,
        help="Generate static UnitMatch report pack from existing artifacts (default: enabled).")
    parser.add_argument("--unitmatch-report-subdir-name", type=str, default="unitmatch_reports",
        help="UnitMatch report output subdirectory under output_dir (default: unitmatch_reports).")
    parser.add_argument("--unitmatch-report-max-heatmap-units", type=int, default=200,
        help="Maximum units rendered in UnitMatch similarity heatmap (default: 200).")
    parser.add_argument("--auto-merge-units", action='store_true',
        help="Optional (default off): run SpikeInterface auto_merge_units during analyzer stage.")
    parser.add_argument("--auto-merge-template-diff-thresh", default="0.05,0.15,0.25",
        help="Comma-separated template_diff_thresh values for auto-merge (used with preset x_contaminations).")
    parser.add_argument("--rerun-analyzer", action='store_true',
        help="Recompute analyzer_output even if checkpoint says complete (does not rerun spikesorting).")

    args = parser.parse_args()
    config = load_config(args.config)
    resolved = resolve_args(args, config)
    fixed_y = resolved["fixed_y"]

    plot_mode = resolved["plot_mode"]
    plot_debug = resolved["plot_debug"]
    raster_sort = resolved["raster_sort"]
    sorter = resolved["sorter"]
    thresholds = resolved["quality_thresholds"]
    rec = args.rec or "rec0000"

    try:
        if args.reanalyze_bursts:
            print("Re-analyzing bursts only on existing spike times...")

        result = run_mea_pipeline(
            MEARunOptions(
                file_path=args.file_path,
                stream_id=args.well,
                recording_num=rec,
                output_root=resolved["output_dir"],
                checkpoint_root=resolved["checkpoint_dir"],
                sorter=sorter,
                docker_image=resolved["docker_image"],
                verbose=bool(args.debug),
                cleanup=bool(resolved["clean_up"]),
                force_restart=bool(args.force_restart),
                resume_from=args.resume_from,
                um_kwargs={
                    "merge_units": bool(args.unitmatch_merge_units),
                    "dry_run": bool(args.unitmatch_dry_run),
                    "scored_dry_run": bool(resolved["unitmatch_scored_dry_run"]),
                    "output_subdir_name": str(resolved["unitmatch_output_subdir_name"]),
                    "throughput_subdir_name": str(resolved["unitmatch_throughput_subdir_name"]),
                    "max_candidate_pairs": int(resolved["unitmatch_max_candidate_pairs"]),
                    "oversplit_min_probability": float(resolved["unitmatch_oversplit_min_probability"]),
                    "oversplit_max_suggestions": int(resolved["unitmatch_oversplit_max_suggestions"]),
                    "apply_merges": bool(resolved["unitmatch_apply_merges"]),
                    "recursive": bool(resolved["unitmatch_recursive"]),
                    "max_iterations": int(resolved["unitmatch_max_iterations"]),
                    "max_spikes_per_unit": int(resolved["unitmatch_max_spikes_per_unit"]),
                    "keep_all_iterations": bool(resolved["unitmatch_keep_all_iterations"]),
                    "generate_reports": bool(args.unitmatch_generate_reports),
                    "report_subdir_name": str(args.unitmatch_report_subdir_name),
                    "report_max_heatmap_units": int(args.unitmatch_report_max_heatmap_units),
                },
                am_kwargs={
                    "enabled": bool(args.auto_merge_units),
                    "template_diff_thresh": str(args.auto_merge_template_diff_thresh),
                },
                option_kwargs={
                    "force_rerun_analyzer": bool(args.rerun_analyzer),
                    "output_subdir_after_well": resolved.get("output_subdir_after_well"),
                },
                reanalyze_bursts=bool(args.reanalyze_bursts),
                skip_spikesorting=bool(args.skip_spikesorting),
                run_analyzer=True,
                run_reports=True,
                thresholds=thresholds,
                no_curation=bool(resolved["no_curation"]),
                export_to_phy=bool(resolved["export_to_phy"]),
                plot_mode=plot_mode,
                plot_debug=bool(plot_debug),
                raster_sort=raster_sort,
                fixed_y=bool(fixed_y),
                auto_merge_template_diff_thresh=str(args.auto_merge_template_diff_thresh),
                extract_rawsortedspikes=bool(args.extract_rawsortedspikes),
            )
        )

        if result.reanalyzed_bursts:
            print("Burst Re-analysis Complete.")
            sys.exit(0)

        if result.skipped:
            sys.exit(0)

        print(f"Processing Complete for {args.well}")

    except Exception as e:
        print(f"CRITICAL FAILURE in {args.well}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
