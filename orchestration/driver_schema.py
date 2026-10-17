"""
Single source of truth for every option ``run_pipeline_driver.py`` accepts.

This schema drives **both**:
  * the web UI form (rendered dynamically from ``/api/schema``), and
  * the watcher's command construction (``build_driver_args``).

Keeping one definition means the UI and the launcher can never drift apart.
Mirrors the argparse definitions in ``run_pipeline_driver.py`` exactly.

Defaults
--------
**Every default here must match ``run_pipeline_driver.py``'s argparse default,
not the effective value.** The driver resolves ``CLI flag -> mea_config.json ->
hardcoded default``, so emitting a flag suppresses the config file. Options the
driver defaults to ``None`` are therefore ``None`` here too, and their effective
value is documented in ``help`` instead. Setting, say, ``sorter`` to
``"kilosort4"`` here would silently override ``sorting.sorter`` in the user's
config file on every run.

Field types
-----------
``flag``    -> ``--name`` emitted only when value is true
``tristate``-> ``--name`` / ``--no-name`` / omitted  (argparse.BooleanOptionalAction)
``str``     -> ``--name VALUE``
``int``     -> ``--name VALUE``
``float``   -> ``--name VALUE``
``choice``  -> ``--name VALUE`` constrained to ``choices``
``list``    -> ``--name V1 V2 ...`` (nargs="+")
``path``    -> like ``str`` but the UI renders a path picker / validator
"""

from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# Each entry: key -> dict(flag, type, default, help, choices?, advanced?)
# ``key`` is the JSON/UI key; ``flag`` is the CLI flag emitted.
SCHEMA: dict[str, list[dict[str, Any]]] = {
    "Input / output": [
        {"key": "config", "flag": "--config", "type": "path", "default": None,
         "help": "Path to config JSON file (CLI flags always override config)"},
        {"key": "output_dir", "flag": "--output-dir", "type": "path", "default": None,
         "help": "Output directory for all results (default: <repo>/AnalyzedData)"},
        {"key": "checkpoint_dir", "flag": "--checkpoint-dir", "type": "path", "default": None,
         "help": "Checkpoint directory (default: <output-dir>/checkpoints)"},
        {"key": "output_subdir_after_well", "flag": "--output-subdir-after-well", "type": "str", "default": None,
         "help": "Optional subdirectory appended under each resolved well output directory"},
        {"key": "export_to_phy", "flag": "--export-to-phy", "type": "flag", "default": False,
         "help": "Export results to Phy format"},
        {"key": "clean_up", "flag": "--clean-up", "type": "flag", "default": False,
         "help": "Remove intermediate files after processing"},
    ],

    "Filtering": [
        {"key": "reference", "flag": "--reference", "type": "path", "default": None,
         "help": "Excel file to filter runs by assay type (needs 'Run #' and 'Assay' columns)"},
        {"key": "type", "flag": "--type", "type": "list", "default": None,
         "help": "Assay types to include (default: 'network today', 'network today/best')"},
    ],

    "Sorting": [
        {"key": "sorter", "flag": "--sorter", "type": "choice", "default": None,
         "choices": ["kilosort4", "mountainsort5", "tridesclous"],
         "help": "Spike sorter to use (default: kilosort4)"},
        {"key": "docker", "flag": "--docker", "type": "str", "default": None,
         "help": "Docker image name for containerized sorting"},
        {"key": "skip_spikesorting", "flag": "--skip-spikesorting", "type": "flag", "default": False,
         "help": "Run spike detection only, skip full sorting"},
        {"key": "extract_rawsortedspikes", "flag": "--extract-rawsortedspikes", "type": "flag", "default": False,
         "help": "Extract per-unit raw mean templates (requires analyzer_output or phy_output)"},
    ],

    "Plotting": [
        {"key": "plot_mode", "flag": "--plot-mode", "type": "choice", "default": None,
         "choices": ["separate", "merged"],
         "help": "Raster and network on separate axes or merged twin-axis (default: separate)"},
        {"key": "raster_sort", "flag": "--raster-sort", "type": "choice", "default": None,
         "choices": ["none", "firing_rate", "location_y", "unit_id"],
         "help": "How to sort units on the raster y-axis (default: none)"},
        {"key": "plot_debug", "flag": "--plot-debug", "type": "flag", "default": False,
         "help": "Overlay burst and superburst intervals on raster plot"},
        {"key": "fixed_y", "flag": "--fixed-y", "type": "flag", "default": False,
         "help": "Fixed y-axis limits for raster plots (run once without it first)"},
    ],

    "Curation": [
        {"key": "no_curation", "flag": "--no-curation", "type": "flag", "default": False,
         "help": "Skip automatic unit curation"},
        {"key": "params", "flag": "--params", "type": "str", "default": None,
         "help": "JSON string or file path with quality thresholds"},
    ],

    "Run control": [
        {"key": "force_restart", "flag": "--force-restart", "type": "flag", "default": False,
         "help": "Ignore checkpoints and restart from scratch"},
        {"key": "reanalyze_bursts", "flag": "--reanalyze-bursts", "type": "flag", "default": False,
         "help": "Re-run burst analysis on existing spike times"},
        {"key": "resume_from", "flag": "--resume-from", "type": "choice", "default": None,
         "choices": ["preprocessing", "sorting", "merge", "analyzer", "reports"],
         "help": "Rewind checkpoint and resume each well from this stage"},
        {"key": "dry", "flag": "--dry", "type": "flag", "default": False,
         "help": "Print what would run without any processing"},
        {"key": "debug", "flag": "--debug", "type": "flag", "default": False,
         "help": "Enable verbose logging"},
    ],

    "UnitMatch (advanced)": [
        {"key": "unitmatch_merge_units", "flag": "--unitmatch-merge-units", "type": "flag", "default": False,
         "help": "Run UnitMatch dry-run/merge phase", "advanced": True},
        {"key": "unitmatch_dry_run", "flag": "--unitmatch-dry-run", "type": "flag", "default": False,
         "help": "Run UnitMatch in dry-run mode", "advanced": True},
        {"key": "unitmatch_scored_dry_run", "flag": "--unitmatch-scored-dry-run", "type": "tristate", "default": None,
         "help": "Attempt backend scoring during dry-run (default: enabled)", "advanced": True},
        {"key": "unitmatch_output_subdir_name", "flag": "--unitmatch-output-subdir-name", "type": "str",
         "default": None, "help": "UnitMatch artifact subdirectory (default: unitmatch_outputs)", "advanced": True},
        {"key": "unitmatch_throughput_subdir_name", "flag": "--unitmatch-throughput-subdir-name", "type": "str",
         "default": None, "help": "UnitMatch throughput subdirectory (default: unitmatch_throughput)", "advanced": True},
        {"key": "unitmatch_max_candidate_pairs", "flag": "--unitmatch-max-candidate-pairs", "type": "int",
         "default": None, "help": "Max candidate pairs (-1 unlimited, 0 none, default: 20000)", "advanced": True},
        {"key": "unitmatch_oversplit_min_probability", "flag": "--unitmatch-oversplit-min-probability",
         "type": "float", "default": None, "help": "Min probability for oversplit suggestions (default: 0.80)",
         "advanced": True},
        {"key": "unitmatch_oversplit_max_suggestions", "flag": "--unitmatch-oversplit-max-suggestions", "type": "int",
         "default": None, "help": "Max oversplit suggestions (-1 unlimited, 0 none, default: 2000)", "advanced": True},
        {"key": "unitmatch_apply_merges", "flag": "--unitmatch-apply-merges", "type": "flag", "default": False,
         "help": "Apply UnitMatch-selected merges instead of report-only", "advanced": True},
        {"key": "unitmatch_recursive", "flag": "--unitmatch-recursive", "type": "flag", "default": False,
         "help": "Recursively rerun merge iterations until convergence", "advanced": True},
        {"key": "unitmatch_max_iterations", "flag": "--unitmatch-max-iterations", "type": "int", "default": None,
         "help": "Max recursive iterations (-1 uncapped, default: 5)", "advanced": True},
        {"key": "unitmatch_max_spikes_per_unit", "flag": "--unitmatch-max-spikes-per-unit", "type": "int",
         "default": None, "help": "Max spikes per unit for raw waveforms (-1 uncapped, default: 100)",
         "advanced": True},
        {"key": "unitmatch_keep_all_iterations", "flag": "--unitmatch-keep-all-iterations", "type": "tristate",
         "default": None, "help": "Keep per-iteration throughput artifacts (default: enabled)", "advanced": True},
        {"key": "unitmatch_generate_reports", "flag": "--unitmatch-generate-reports", "type": "tristate",
         "default": None, "help": "Generate static UnitMatch report pack (default: enabled)", "advanced": True},
        {"key": "unitmatch_report_subdir_name", "flag": "--unitmatch-report-subdir-name", "type": "str",
         "default": None, "help": "UnitMatch report subdirectory (default: unitmatch_reports)", "advanced": True},
        {"key": "unitmatch_report_max_heatmap_units", "flag": "--unitmatch-report-max-heatmap-units", "type": "int",
         "default": None, "help": "Max units in similarity heatmap (default: 200)", "advanced": True},
    ],
}


# Flat lookup: key -> field spec
FIELDS: dict[str, dict[str, Any]] = {
    f["key"]: {**f, "group": group}
    for group, fields in SCHEMA.items()
    for f in fields
}


def default_options() -> dict[str, Any]:
    """Return a dict of every option at its default value."""
    return {key: spec.get("default") for key, spec in FIELDS.items()}


def build_driver_args(options: dict[str, Any]) -> list[str]:
    """Translate a UI/JSON options dict into argv for run_pipeline_driver.py.

    Only values that differ from "unset" are emitted, so the driver's own
    CLI -> config -> default priority chain is preserved.
    """
    argv: list[str] = []

    for key, spec in FIELDS.items():
        if key not in options:
            continue
        value = options[key]
        ftype = spec["type"]
        flag = spec["flag"]

        if value is None or value == "":
            continue

        if ftype == "flag":
            if value is True:
                argv.append(flag)

        elif ftype == "tristate":
            if value is True:
                argv.append(flag)
            elif value is False:
                argv.append("--no-" + flag.lstrip("-"))

        elif ftype == "list":
            seq = value if isinstance(value, (list, tuple)) else [value]
            seq = [str(v) for v in seq if str(v).strip()]
            if seq:
                argv.append(flag)
                argv.extend(seq)

        else:  # str / path / int / float / choice
            argv.extend([flag, str(value)])

    return argv


def validate_options(options: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    for key, value in options.items():
        spec = FIELDS.get(key)
        if spec is None:
            errors.append(f"Unknown option: {key}")
            continue
        if value is None or value == "":
            continue
        ftype = spec["type"]
        if ftype == "choice" and value not in spec.get("choices", []):
            errors.append(f"{key}: '{value}' is not one of {spec['choices']}")
        elif ftype == "int":
            try:
                int(value)
            except (TypeError, ValueError):
                errors.append(f"{key}: expected an integer, got '{value}'")
        elif ftype == "float":
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append(f"{key}: expected a number, got '{value}'")
    return errors


def schema_for_ui() -> list[dict[str, Any]]:
    """Serializable schema for the frontend form renderer."""
    return [
        {"group": group, "fields": [{k: v for k, v in f.items()} for f in fields]}
        for group, fields in SCHEMA.items()
    ]
