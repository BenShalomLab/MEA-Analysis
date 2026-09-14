# ==========================================================
# config_loader.py
# Shared config loader for run_pipeline_driver.py and mea_analysis_routine.py
# Priority: CLI arg -> config file -> hardcoded default
# ==========================================================

import os
import json
from pathlib import Path

# ----------------------------------------------------------
# Hardcoded Defaults
# ----------------------------------------------------------
DEFAULTS = {
    "io": {
        "output_dir":       None,
        "checkpoint_dir":   None,
        "output_subdir_after_well": None,
        # CSV/TSV/XLSX describing what was cultured in each well: genotype,
        # line, prep_type, batch, plating_date. See docs/samples_schema.md.
        "samples_file":     None,
        "export_to_phy":    False,
        "clean_up":         False,
    },
    "sorting": {
        "sorter":           "kilosort4",
        "docker_image":     None,
        # Explicit Kilosort4 parameter overrides, merged over the built-in
        # VRAM-tier defaults. Pin the result-affecting ones here (dmin,
        # cluster_downsampling, max_cluster_subset, invert_sign) for any
        # dataset processed on more than one GPU model.
        "kilosort_params":  None,
    },
    "burst_detection": {
        "burst_detector":   "parameter_free",
        "gaussian_bin_size_s":            0.01,
        "gaussian_sigma_s":               0.1,
        "gaussian_min_prominence":        None,
        "gaussian_min_peak_distance_s":   1.0,
        "gaussian_onset_offset_peak_frac": 0.3,
        "gaussian_min_height_sd":         2.0,
        "parameter_free_min_superburst_dur_s":      2.5,
        "parameter_free_min_superburst_components": 2,
    },
    "connectivity": {
        # Pairwise STTC plus graph topology. Cost is quadratic in the unit
        # count, hence max_units; above it the matrix is computed on a random
        # subsample and the result says so. Measured on ~2000 spikes per unit
        # over a 5 minute recording: about 10 s per well at 200 units and
        # about 45 s at 400, for both windows together. Set enabled to false
        # (or pass --no-connectivity) to skip it entirely.
        "enabled":        True,
        "max_units":      400,
        "windows_s":      [0.010, 0.025],
        "n_null_pairs":   500,
        "n_null_shifts":  20,
        "percentile":     95.0,
    },
    "merging": {
        "unitmatch_scored_dry_run": True,
        "unitmatch_output_subdir_name": "unitmatch_outputs",
        "unitmatch_throughput_subdir_name": "unitmatch_throughput",
        "unitmatch_max_candidate_pairs": 20000,
        "unitmatch_oversplit_min_probability": 0.80,
        "unitmatch_oversplit_max_suggestions": 2000,
        "unitmatch_apply_merges": False,
        "unitmatch_recursive": False,
        "unitmatch_max_iterations": 5,
        "unitmatch_max_spikes_per_unit": 100,
        "unitmatch_keep_all_iterations": False,
        "unitmatch_generate_reports": True,
        "unitmatch_report_subdir_name": "unitmatch_reports",
        "unitmatch_report_max_heatmap_units": 200,
    },
    "filtering": {
        "reference_file":   None,
        "assay_types":      ["network today", "network today/best"],
    },
    "plotting": {
        "plot_mode":        "separate",
        "raster_sort":      "none",
        "plot_debug":       False,
        "fixed_y":       False,
    },
    "curation": {
        "no_curation":      False,
        "quality_thresholds": {
            "presence_ratio":       0.75,
            "rp_contamination":     0.15,
            "firing_rate":          0.05,
            "amplitude_median":     -20,
            "amplitude_cv_median":  0.5,
        }
    }
}


# ----------------------------------------------------------
# Load config file
# ----------------------------------------------------------
def load_config(config_path=None):
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


# ----------------------------------------------------------
# Resolve a single value: CLI -> config -> default
# ----------------------------------------------------------
def _resolve(cli_val, cfg_val, default):
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default


def _cfg(config, section, key):
    return config.get(section, {}).get(key, None)


# ----------------------------------------------------------
# Thresholds need deep merge: defaults -> config -> CLI --params
# ----------------------------------------------------------
def _resolve_thresholds(args, config):
    thresholds = DEFAULTS["curation"]["quality_thresholds"].copy()
    thresholds.update(config.get("curation", {}).get("quality_thresholds", {}))
    cli_params = getattr(args, "params", None)
    if cli_params:
        if os.path.exists(cli_params):
            with open(cli_params) as f:
                thresholds.update(json.load(f))
        else:
            thresholds.update(json.loads(cli_params))
    return thresholds


# ----------------------------------------------------------
# Kilosort parameter overrides: config dict, then CLI JSON/path, merged
# ----------------------------------------------------------
def _resolve_kilosort_params(args, config):
    params = {}
    cfg_params = _cfg(config, "sorting", "kilosort_params")
    if isinstance(cfg_params, dict):
        params.update({k: v for k, v in cfg_params.items() if not k.startswith("_")})

    cli_params = getattr(args, "kilosort_params", None)
    if cli_params:
        if os.path.exists(cli_params):
            with open(cli_params) as f:
                params.update(json.load(f))
        else:
            params.update(json.loads(cli_params))
    return params


# ----------------------------------------------------------
# Bool store_true flags: only True if explicitly passed
# False from argparse means "not passed" so return None
# ----------------------------------------------------------
def _bool(args, attr):
    return True if getattr(args, attr, None) is True else None


# ----------------------------------------------------------
# Main resolver - works for both scripts
# ----------------------------------------------------------
def resolve_args(args, config):
    return {
        # io
        "output_dir":                   _resolve(getattr(args, "output_dir", None),                 _cfg(config, "io", "output_dir"),               DEFAULTS["io"]["output_dir"]),
        "checkpoint_dir":               _resolve(getattr(args, "checkpoint_dir", None),             _cfg(config, "io", "checkpoint_dir"),           DEFAULTS["io"]["checkpoint_dir"]),
        "output_subdir_after_well":     _resolve(getattr(args, "output_subdir_after_well", None),   _cfg(config, "io", "output_subdir_after_well"), DEFAULTS["io"]["output_subdir_after_well"]),
        "samples_file":                 _resolve(getattr(args, "samples_file", None),               _cfg(config, "io", "samples_file"),             DEFAULTS["io"]["samples_file"]),
        "export_to_phy":    _resolve(_bool(args, "export_to_phy"),           _cfg(config, "io", "export_to_phy"),         DEFAULTS["io"]["export_to_phy"]),
        "clean_up":         _resolve(_bool(args, "clean_up"),                _cfg(config, "io", "clean_up"),              DEFAULTS["io"]["clean_up"]),
        # sorting
        "sorter":           _resolve(getattr(args, "sorter", None),          _cfg(config, "sorting", "sorter"),           DEFAULTS["sorting"]["sorter"]),
        "docker_image":     _resolve(getattr(args, "docker", None),          _cfg(config, "sorting", "docker_image"),     DEFAULTS["sorting"]["docker_image"]),
        "kilosort_params":  _resolve_kilosort_params(args, config),
        # burst detection
        "burst_detector":   _resolve(getattr(args, "burst_detector", None),  _cfg(config, "burst_detection", "burst_detector"), DEFAULTS["burst_detection"]["burst_detector"]),
        "gaussian_bin_size_s":             _resolve(getattr(args, "gaussian_bin_size_s", None),             _cfg(config, "burst_detection", "gaussian_bin_size_s"),             DEFAULTS["burst_detection"]["gaussian_bin_size_s"]),
        "gaussian_sigma_s":                _resolve(getattr(args, "gaussian_sigma_s", None),                _cfg(config, "burst_detection", "gaussian_sigma_s"),                DEFAULTS["burst_detection"]["gaussian_sigma_s"]),
        "gaussian_min_prominence":         _resolve(getattr(args, "gaussian_min_prominence", None),         _cfg(config, "burst_detection", "gaussian_min_prominence"),         DEFAULTS["burst_detection"]["gaussian_min_prominence"]),
        "gaussian_min_peak_distance_s":    _resolve(getattr(args, "gaussian_min_peak_distance_s", None),    _cfg(config, "burst_detection", "gaussian_min_peak_distance_s"),    DEFAULTS["burst_detection"]["gaussian_min_peak_distance_s"]),
        "gaussian_onset_offset_peak_frac": _resolve(getattr(args, "gaussian_onset_offset_peak_frac", None), _cfg(config, "burst_detection", "gaussian_onset_offset_peak_frac"), DEFAULTS["burst_detection"]["gaussian_onset_offset_peak_frac"]),
        "gaussian_min_height_sd":          _resolve(getattr(args, "gaussian_min_height_sd", None),          _cfg(config, "burst_detection", "gaussian_min_height_sd"),          DEFAULTS["burst_detection"]["gaussian_min_height_sd"]),
        "parameter_free_min_superburst_dur_s":      _resolve(getattr(args, "parameter_free_min_superburst_dur_s", None),      _cfg(config, "burst_detection", "parameter_free_min_superburst_dur_s"),      DEFAULTS["burst_detection"]["parameter_free_min_superburst_dur_s"]),
        "parameter_free_min_superburst_components": _resolve(getattr(args, "parameter_free_min_superburst_components", None), _cfg(config, "burst_detection", "parameter_free_min_superburst_components"), DEFAULTS["burst_detection"]["parameter_free_min_superburst_components"]),
        # connectivity
        "connectivity_enabled": (
            False if getattr(args, "no_connectivity", False)
            else _resolve(None, _cfg(config, "connectivity", "enabled"), DEFAULTS["connectivity"]["enabled"])
        ),
        "connectivity_max_units":     _resolve(getattr(args, "connectivity_max_units", None), _cfg(config, "connectivity", "max_units"),     DEFAULTS["connectivity"]["max_units"]),
        "connectivity_windows_s":     _resolve(None,                                          _cfg(config, "connectivity", "windows_s"),     DEFAULTS["connectivity"]["windows_s"]),
        "connectivity_n_null_pairs":  _resolve(None,                                          _cfg(config, "connectivity", "n_null_pairs"),  DEFAULTS["connectivity"]["n_null_pairs"]),
        "connectivity_n_null_shifts": _resolve(None,                                          _cfg(config, "connectivity", "n_null_shifts"), DEFAULTS["connectivity"]["n_null_shifts"]),
        "connectivity_percentile":    _resolve(None,                                          _cfg(config, "connectivity", "percentile"),    DEFAULTS["connectivity"]["percentile"]),
        # merging (UnitMatch)
        "unitmatch_scored_dry_run":             _resolve(_bool(args, "unitmatch_scored_dry_run"),                       _cfg(config, "merging", "unitmatch_scored_dry_run"),                DEFAULTS["merging"]["unitmatch_scored_dry_run"]),
        "unitmatch_output_subdir_name":         _resolve(getattr(args, "unitmatch_output_subdir_name", None),           _cfg(config, "merging", "unitmatch_output_subdir_name"),            DEFAULTS["merging"]["unitmatch_output_subdir_name"]),
        "unitmatch_throughput_subdir_name":     _resolve(getattr(args, "unitmatch_throughput_subdir_name", None),       _cfg(config, "merging", "unitmatch_throughput_subdir_name"),        DEFAULTS["merging"]["unitmatch_throughput_subdir_name"]),
        "unitmatch_max_candidate_pairs":        _resolve(getattr(args, "unitmatch_max_candidate_pairs", None),          _cfg(config, "merging", "unitmatch_max_candidate_pairs"),           DEFAULTS["merging"]["unitmatch_max_candidate_pairs"]),
        "unitmatch_oversplit_min_probability":  _resolve(getattr(args, "unitmatch_oversplit_min_probability", None),    _cfg(config, "merging", "unitmatch_oversplit_min_probability"),     DEFAULTS["merging"]["unitmatch_oversplit_min_probability"]),
        "unitmatch_oversplit_max_suggestions":  _resolve(getattr(args, "unitmatch_oversplit_max_suggestions", None),    _cfg(config, "merging", "unitmatch_oversplit_max_suggestions"),     DEFAULTS["merging"]["unitmatch_oversplit_max_suggestions"]),
        "unitmatch_apply_merges":               _resolve(_bool(args, "unitmatch_apply_merges"),                         _cfg(config, "merging", "unitmatch_apply_merges"),                  DEFAULTS["merging"]["unitmatch_apply_merges"]),
        "unitmatch_recursive":                  _resolve(_bool(args, "unitmatch_recursive"),                            _cfg(config, "merging", "unitmatch_recursive"),                     DEFAULTS["merging"]["unitmatch_recursive"]),
        "unitmatch_max_iterations":             _resolve(getattr(args, "unitmatch_max_iterations", None),               _cfg(config, "merging", "unitmatch_max_iterations"),                DEFAULTS["merging"]["unitmatch_max_iterations"]),
        "unitmatch_max_spikes_per_unit":        _resolve(getattr(args, "unitmatch_max_spikes_per_unit", None),         _cfg(config, "merging", "unitmatch_max_spikes_per_unit"),           DEFAULTS["merging"]["unitmatch_max_spikes_per_unit"]),
        "unitmatch_keep_all_iterations":        _resolve(getattr(args, "unitmatch_keep_all_iterations", None),          _cfg(config, "merging", "unitmatch_keep_all_iterations"),           DEFAULTS["merging"]["unitmatch_keep_all_iterations"]),
        "unitmatch_generate_reports":           _resolve(_bool(args, "unitmatch_generate_reports"),                     _cfg(config, "merging", "unitmatch_generate_reports"),              DEFAULTS["merging"]["unitmatch_generate_reports"]),
        "unitmatch_report_subdir_name":         _resolve(getattr(args, "unitmatch_report_subdir_name", None),           _cfg(config, "merging", "unitmatch_report_subdir_name"),            DEFAULTS["merging"]["unitmatch_report_subdir_name"]),
        "unitmatch_report_max_heatmap_units":   _resolve(getattr(args, "unitmatch_report_max_heatmap_units", None),     _cfg(config, "merging", "unitmatch_report_max_heatmap_units"),      DEFAULTS["merging"]["unitmatch_report_max_heatmap_units"]),
        # filtering (driver only - ignored silently in routine)
        "reference_file":   _resolve(getattr(args, "reference", None),       _cfg(config, "filtering", "reference_file"), DEFAULTS["filtering"]["reference_file"]),
        "assay_types":      _resolve(getattr(args, "type", None),            _cfg(config, "filtering", "assay_types"),    DEFAULTS["filtering"]["assay_types"]),
        # plotting
        "plot_mode":        _resolve(getattr(args, "plot_mode", None),       _cfg(config, "plotting", "plot_mode"),       DEFAULTS["plotting"]["plot_mode"]),
        "raster_sort":      _resolve(getattr(args, "raster_sort", None),     _cfg(config, "plotting", "raster_sort"),     DEFAULTS["plotting"]["raster_sort"]),
        "plot_debug":       _resolve(_bool(args, "plot_debug"),              _cfg(config, "plotting", "plot_debug"),      DEFAULTS["plotting"]["plot_debug"]),
        "fixed_y":          _resolve(_bool(args, "fixed_y"),                 _cfg(config, "plotting", "fixed_y"),         DEFAULTS["plotting"].get("fixed_y", False)),
        # curation
        "no_curation":          _resolve(_bool(args, "no_curation"),         _cfg(config, "curation", "no_curation"),     DEFAULTS["curation"]["no_curation"]),
        "quality_thresholds":   _resolve_thresholds(args, config),
    }


# ----------------------------------------------------------
# Build extra_arg_string for driver subprocess
# Receives resolved dict + raw args for CLI-only pass-throughs
# ----------------------------------------------------------
def build_extra_args(resolved, cli_args):
    extra = []

    # Pass config downstream so subprocess inherits it
    if getattr(cli_args, "config", None):
        extra.append(f"--config '{cli_args.config}'")

    # io
    if resolved["output_dir"]:      extra.append(f"--output-dir '{resolved['output_dir']}'")
    if resolved["checkpoint_dir"]:  extra.append(f"--checkpoint-dir '{resolved['checkpoint_dir']}'")
    if resolved["output_subdir_after_well"]:
        extra.append(f"--output-subdir-after-well '{resolved['output_subdir_after_well']}'")
    if resolved.get("samples_file"):
        extra.append(f"--samples-file '{resolved['samples_file']}'")
    if resolved["export_to_phy"]:   extra.append("--export-to-phy")
    if resolved["clean_up"]:        extra.append("--clean-up")

    # sorting
    if resolved["sorter"]:          extra.append(f"--sorter {resolved['sorter']}")
    if resolved["docker_image"]:    extra.append(f"--docker {resolved['docker_image']}")
    if resolved.get("kilosort_params"):
        # JSON never contains a single quote, so single-quoting is safe here.
        extra.append(f"--kilosort-params '{json.dumps(resolved['kilosort_params'])}'")

    # burst detection
    if resolved["burst_detector"]:  extra.append(f"--burst-detector {resolved['burst_detector']}")
    if resolved.get("gaussian_bin_size_s") is not None:
        extra.append(f"--gaussian-bin-size-s {resolved['gaussian_bin_size_s']}")
    if resolved.get("gaussian_sigma_s") is not None:
        extra.append(f"--gaussian-sigma-s {resolved['gaussian_sigma_s']}")
    if resolved.get("gaussian_min_prominence") is not None:
        extra.append(f"--gaussian-min-prominence {resolved['gaussian_min_prominence']}")
    if resolved.get("gaussian_min_peak_distance_s") is not None:
        extra.append(f"--gaussian-min-peak-distance-s {resolved['gaussian_min_peak_distance_s']}")
    if resolved.get("gaussian_onset_offset_peak_frac") is not None:
        extra.append(f"--gaussian-onset-offset-peak-frac {resolved['gaussian_onset_offset_peak_frac']}")
    if resolved.get("gaussian_min_height_sd") is not None:
        extra.append(f"--gaussian-min-height-sd {resolved['gaussian_min_height_sd']}")
    if resolved.get("parameter_free_min_superburst_dur_s") is not None:
        extra.append(f"--pf-min-superburst-dur-s {resolved['parameter_free_min_superburst_dur_s']}")
    if resolved.get("parameter_free_min_superburst_components") is not None:
        extra.append(f"--pf-min-superburst-components {int(resolved['parameter_free_min_superburst_components'])}")

    # connectivity
    if not resolved.get("connectivity_enabled", True):
        extra.append("--no-connectivity")
    if resolved.get("connectivity_max_units") is not None:
        extra.append(f"--connectivity-max-units {int(resolved['connectivity_max_units'])}")

    # plotting
    if resolved["plot_mode"]:       extra.append(f"--plot-mode {resolved['plot_mode']}")
    if resolved["raster_sort"]:     extra.append(f"--raster-sort {resolved['raster_sort']}")
    if resolved["plot_debug"]:      extra.append("--plot-debug")
    if resolved["fixed_y"]:         extra.append("--fixed-y")

    # curation
    if resolved["no_curation"]:     extra.append("--no-curation")

    # CLI-only flags - passed through directly, never in config
    if getattr(cli_args, "force_restart", False):     extra.append("--force-restart")
    if getattr(cli_args, "debug", False):             extra.append("--debug")
    if getattr(cli_args, "skip_spikesorting", False): extra.append("--skip-spikesorting")
    if getattr(cli_args, "extract_rawsortedspikes", False): extra.append("--extract-rawsortedspikes")
    if getattr(cli_args, "reanalyze_bursts", False):  extra.append("--reanalyze-bursts")
    if getattr(cli_args, "resume_from", None):         extra.append(f"--resume-from {getattr(cli_args, 'resume_from')}")
    if getattr(cli_args, "unitmatch_merge_units", False): extra.append("--unitmatch-merge-units")
    if getattr(cli_args, "unitmatch_dry_run", False):     extra.append("--unitmatch-dry-run")

    # merging (UnitMatch)
    unitmatch_scored = getattr(cli_args, "unitmatch_scored_dry_run", None)
    if unitmatch_scored is True:
        extra.append("--unitmatch-scored-dry-run")
    elif unitmatch_scored is False:
        extra.append("--no-unitmatch-scored-dry-run")
    if getattr(cli_args, "unitmatch_output_subdir_name", None):
        extra.append(f"--unitmatch-output-subdir-name '{getattr(cli_args, 'unitmatch_output_subdir_name')}'")
    if getattr(cli_args, "unitmatch_throughput_subdir_name", None):
        extra.append(
            f"--unitmatch-throughput-subdir-name '{getattr(cli_args, 'unitmatch_throughput_subdir_name')}'"
        )
    if getattr(cli_args, "unitmatch_max_candidate_pairs", None) is not None:
        extra.append(f"--unitmatch-max-candidate-pairs {int(getattr(cli_args, 'unitmatch_max_candidate_pairs'))}")
    if getattr(cli_args, "unitmatch_oversplit_min_probability", None) is not None:
        extra.append(
            f"--unitmatch-oversplit-min-probability {float(getattr(cli_args, 'unitmatch_oversplit_min_probability'))}"
        )
    if getattr(cli_args, "unitmatch_oversplit_max_suggestions", None) is not None:
        extra.append(
            f"--unitmatch-oversplit-max-suggestions {int(getattr(cli_args, 'unitmatch_oversplit_max_suggestions'))}"
        )
    if getattr(cli_args, "unitmatch_apply_merges", False):
        extra.append("--unitmatch-apply-merges")
    if getattr(cli_args, "unitmatch_recursive", False):
        extra.append("--unitmatch-recursive")
    if getattr(cli_args, "unitmatch_max_iterations", None) is not None:
        extra.append(f"--unitmatch-max-iterations {int(getattr(cli_args, 'unitmatch_max_iterations'))}")
    if getattr(cli_args, "unitmatch_max_spikes_per_unit", None) is not None:
        extra.append(f"--unitmatch-max-spikes-per-unit {int(getattr(cli_args, 'unitmatch_max_spikes_per_unit'))}")
    keep_all_iters = getattr(cli_args, "unitmatch_keep_all_iterations", None)
    if keep_all_iters is True:
        extra.append("--unitmatch-keep-all-iterations")
    elif keep_all_iters is False:
        extra.append("--no-unitmatch-keep-all-iterations")
    generate_reports = getattr(cli_args, "unitmatch_generate_reports", None)
    if generate_reports is True:
        extra.append("--unitmatch-generate-reports")
    elif generate_reports is False:
        extra.append("--no-unitmatch-generate-reports")
    if getattr(cli_args, "unitmatch_report_subdir_name", None):
        extra.append(f"--unitmatch-report-subdir-name '{getattr(cli_args, 'unitmatch_report_subdir_name')}'")
    if getattr(cli_args, "unitmatch_report_max_heatmap_units", None) is not None:
        extra.append(
            f"--unitmatch-report-max-heatmap-units {int(getattr(cli_args, 'unitmatch_report_max_heatmap_units'))}"
        )

    return " ".join(extra)


# ----------------------------------------------------------
# Template generator - run directly to get a starter config
# python config_loader.py [optional_output_path]
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "mea_config_template.json"
    with open(output, "w") as f:
        json.dump(DEFAULTS, f, indent=2)
    print(f"Template written to: {output}")
