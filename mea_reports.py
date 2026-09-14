import os
import re
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from matplotlib.lines import Line2D
import spikeinterface.full as si

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage

try:
    from mea_infra import collect_provenance
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_infra import collect_provenance

try:
    from unit_bursts import compute_unit_burst_features
except ImportError:
    try:
        from MEA_Analysis.IPNAnalysis.unit_bursts import compute_unit_burst_features
    except ImportError:
        compute_unit_burst_features = None

try:
    from parameter_free_burst_detector import compute_network_bursts as compute_network_bursts_parameter_free
    from gaussianNetworkBursts import compute_network_bursts as compute_network_bursts_gaussian
    import helper_functions as helper
    from scalebury import add_scalebar
except ImportError:
    try:
        from MEA_Analysis.IPNAnalysis.parameter_free_burst_detector import compute_network_bursts as compute_network_bursts_parameter_free
        from MEA_Analysis.IPNAnalysis.gaussianNetworkBursts import compute_network_bursts as compute_network_bursts_gaussian
        from MEA_Analysis.IPNAnalysis import helper_functions as helper
        from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar
    except ImportError:
        compute_network_bursts_parameter_free = None
        compute_network_bursts_gaussian = None
        helper = None
        add_scalebar = None

# Registry of interchangeable network-burst detectors. Both return the same
# schema (see parameter_free_burst_detector.compute_network_bursts docstring
# and gaussianNetworkBursts.compute_network_bursts docstring); the Gaussian
# detector only populates the "network_bursts" tier ("burst_fragments" and
# "superbursts" come back empty — it does not do fragment/superburst merging).
BURST_DETECTORS = {
    "parameter_free": compute_network_bursts_parameter_free,
    "gaussian": compute_network_bursts_gaussian,
}


class ReportsMixin:
    """Phase 4: curation, waveform PDF, probe location plots, and burst analysis."""

    def generate_reports(self, thresholds=None, no_curation=False, export_phy=False,
                         plot_mode="separate", plot_debug=False, raster_sort=None, fixed_y=False):
        if self.state['stage'] == ProcessingStage.REPORTS_COMPLETE.value:
            return

        self.logger.info("--- [Phase 4] Reports & Curation ---")
        try:
            q_metrics = self.analyzer.get_extension("quality_metrics").get_data()
            t_metrics = self.analyzer.get_extension("template_metrics").get_data()
            locations = self.analyzer.get_extension("unit_locations").get_data()

            q_metrics['loc_x'] = locations[:, 0]
            q_metrics['loc_y'] = locations[:, 1]

            q_metrics.to_excel(self.output_dir / "qm_unfiltered.xlsx")
            t_metrics.to_excel(self.output_dir / "tm_unfiltered.xlsx")
            self._plot_probe_locations(q_metrics.index.values, locations, "locations_unfiltered.pdf")

            if no_curation:
                self.logger.info("Skipping curation.")
                clean_units = q_metrics.index.values
                self.curation_summary = {
                    "applied": False,
                    "n_units_input": int(len(q_metrics)),
                    "n_units_kept": int(len(q_metrics)),
                    "n_units_rejected": 0,
                    "rejected_by_reason": {},
                }
            else:
                self.logger.info("Applying curation.")
                clean_metrics, rejection_log = self._apply_curation_logic(q_metrics, thresholds)
                clean_units = clean_metrics.index.values
                clean_metrics.to_excel(self.output_dir / "metrics_curated.xlsx")
                rejection_log.to_excel(self.output_dir / "rejection_log.xlsx")
                t_metrics.loc[clean_units].to_excel(self.output_dir / "tm_curated.xlsx")

            with open(self.output_dir / "curation_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.curation_summary, f, indent=2)

            if len(clean_units) == 0:
                self.logger.warning("No units passed curation.")
                self._save_checkpoint(ProcessingStage.REPORTS_COMPLETE, n_units=0,
                                      curation=self.curation_summary)
                return

            mask = np.isin(self.analyzer.unit_ids, clean_units)
            self._plot_probe_locations(clean_units, locations[mask], f"locations_{len(clean_units)}_units.pdf")
            self._plot_waveforms_grid(clean_units)
            self._run_burst_analysis(clean_units, plot_mode=plot_mode, plot_debug=plot_debug,
                                     raster_sort=raster_sort, fixed_y=fixed_y)

            if export_phy:
                phy_folder = self.output_dir / "phy_output"
                si.export_to_phy(self.analyzer.select_units(clean_units),
                                 output_folder=phy_folder,
                                 remove_if_exists=True, copy_binary=False)
                self._patch_phy_binary_path(phy_folder)

            self._save_checkpoint(ProcessingStage.REPORTS_COMPLETE, n_units=len(clean_units),
                                  failed_stage=None, error=None,
                                  curation=self.curation_summary)
        except Exception as e:
            err = {
                "failed_stage": ProcessingStage.REPORTS.name,
                "exception": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "time": str(datetime.now())
            }
            self.logger.error(err["traceback"])
            self._save_checkpoint(ProcessingStage.ANALYZER_COMPLETE, error=err)
            raise

    def _apply_curation_logic(self, metrics, user_thresholds):
        """Threshold-based unit curation.

        Also records, in self.curation_summary, how many units each rule
        rejected. Curation is not neutral: a rule like presence_ratio removes
        units that fire only inside bursts, which are exactly the units a
        hypoactive genotype has most of. Without per-reason counts that bias
        is invisible, so the counts are written per well and should be
        compared across conditions before any group statistics are trusted.
        """
        defaults = {'presence_ratio': 0.75, 'rp_contamination': 0.15, 'firing_rate': 0.05,
                    'firing_rate_max': 100.0, 'amplitude_median': -20, 'amplitude_cv_median': 0.5}
        if user_thresholds:
            defaults.update(user_thresholds)

        # (label, metric column, predicate on the metric value -> reject?)
        # amplitude_median is compared in absolute value: SpikeInterface has
        # reported it both signed (negative, peak_sign='neg') and as a
        # magnitude across versions, so comparing the raw signed value against
        # a negative threshold silently inverts the rule on the other
        # convention. The intent is "reject units whose spikes are too small".
        rules = [
            ("Low Presence", 'presence_ratio',
             lambda v: v < defaults['presence_ratio']),
            ("High Contam", 'rp_contamination',
             lambda v: v > defaults['rp_contamination']),
            ("Low FR", 'firing_rate',
             lambda v: v < defaults['firing_rate']),
            # Physiologically-implausible rate (>100 Hz sustained) usually means a
            # noisy/faulty electrode rather than a real unit; Kilosort4 doesn't
            # always cluster these out.
            ("Implausibly High FR", 'firing_rate',
             lambda v: v > defaults['firing_rate_max']),
            ("Low Amp", 'amplitude_median',
             lambda v: abs(v) < abs(defaults['amplitude_median'])),
            ("Unstable Amp", 'amplitude_cv_median',
             lambda v: v > defaults['amplitude_cv_median']),
        ]

        available = set(metrics.columns)
        skipped_rules = sorted({col for _, col, _ in rules if col not in available})
        if skipped_rules:
            self.logger.warning(
                "Curation: metric(s) %s not present in this analyzer's quality metrics; "
                "the corresponding rule(s) were not applied.", skipped_rules,
            )

        keep_mask = np.ones(len(metrics), dtype=bool)
        rejections = []
        reason_counts = {}
        n_missing_values = {}

        for position, (unit_id, row) in enumerate(metrics.iterrows()):
            reasons = []
            for label, column, is_rejected in rules:
                if column not in available:
                    continue
                value = row[column]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    n_missing_values[column] = n_missing_values.get(column, 0) + 1
                    continue
                if is_rejected(float(value)):
                    reasons.append(label)

            if reasons:
                keep_mask[position] = False
                rejections.append({"unit_id": unit_id, "reasons": "; ".join(reasons)})
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        self.curation_summary = {
            "applied": True,
            "n_units_input": int(len(metrics)),
            "n_units_kept": int(keep_mask.sum()),
            "n_units_rejected": int((~keep_mask).sum()),
            # A unit failing several rules is counted once per rule, so these
            # sum to >= n_units_rejected.
            "rejected_by_reason": reason_counts,
            "thresholds": {k: defaults[k] for k in sorted(defaults)},
            "rules_skipped_missing_metric": skipped_rules,
            "units_with_missing_metric": n_missing_values,
        }
        self.logger.info(
            "Curation: kept %d/%d units; rejections by reason: %s",
            self.curation_summary["n_units_kept"],
            self.curation_summary["n_units_input"],
            reason_counts or "none",
        )

        return metrics[keep_mask], pd.DataFrame(rejections)

    def _plot_probe_locations(self, unit_ids, locations, filename):
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        si.plot_probe_map(self.recording, ax=ax, with_channel_ids=False)
        ax.scatter(locations[:, 0], locations[:, 1], s=10, c='blue', alpha=0.6)
        ax.invert_yaxis()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)

    def _plot_waveforms_grid(self, unit_ids):
        pdf_path = self.output_dir / "waveforms_grid.pdf"
        self.logger.info(f"Generating PDF: {pdf_path}")

        wf_ext = self.analyzer.get_extension("waveforms")
        fs = self.recording.get_sampling_frequency()

        with pdf.PdfPages(pdf_path) as pdf_doc:
            units_per_page = 12
            for i in range(0, len(unit_ids), units_per_page):
                batch = unit_ids[i : i + units_per_page]
                fig, axes = plt.subplots(3, 4, figsize=(12, 9))
                axes = axes.flatten()

                for ax, uid in zip(axes, batch):
                    wf = wf_ext.get_waveforms_one_unit(uid)
                    mean_wf = np.mean(wf, axis=0)
                    best_ch = np.argmin(np.min(mean_wf, axis=0))

                    time_ms = np.arange(wf.shape[1]) / fs * 1000

                    n_spikes = wf.shape[0]
                    if n_spikes > 100:
                        indices = np.random.choice(n_spikes, 100, replace=False)
                        spikes_to_plot = wf[indices, :, best_ch]
                    else:
                        spikes_to_plot = wf[:, :, best_ch]

                    ax.plot(time_ms, spikes_to_plot.T, c='gray', lw=0.5, alpha=0.3)
                    ax.plot(time_ms, mean_wf[:, best_ch], c='red', lw=1.5)
                    ax.set_title(f"Unit {uid} | Ch {best_ch}", fontsize=10)

                    try:
                        add_scalebar(ax,
                                     matchx=False, matchy=False,
                                     sizex=1.0, labelx='1 ms',
                                     sizey=50, labely='50 µV',
                                     loc='lower right',
                                     hidex=True, hidey=True)
                    except Exception:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                for j in range(len(batch), len(axes)):
                    axes[j].axis('off')

                pdf_doc.savefig(fig)
                plt.close(fig)

    @staticmethod
    def _build_unit_stats_frame(detector_unit_stats, per_unit_bursts):
        """One row per unit: detector ISI statistics plus burst features.

        Both sources report mean_firing_rate_hz; the duplicate is dropped so
        the CSV does not end up with two identically named columns, which
        pandas will happily write and then refuse to index by name.
        """
        frames = []
        if detector_unit_stats:
            frames.append(pd.DataFrame.from_dict(detector_unit_stats, orient="index"))
        if per_unit_bursts:
            burst_frame = pd.DataFrame.from_dict(per_unit_bursts, orient="index")
            if frames:
                duplicates = [c for c in burst_frame.columns if c in frames[0].columns]
                burst_frame = burst_frame.drop(columns=duplicates)
            frames.append(burst_frame)

        if not frames:
            return None

        frame = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
        frame.index.name = "unit_id"
        return frame

    def _compute_unit_level_features(self, spike_times, duration_s):
        """Per-unit burst features for every unit, or {} if unavailable.

        Never fatal: this is an added measurement, and losing it should not
        cost a well its network results after a long sort.
        """
        if compute_unit_burst_features is None:
            self.logger.warning(
                "unit_bursts module unavailable; per-unit burst features skipped."
            )
            return {}

        try:
            result = compute_unit_burst_features(spike_times, duration_s=duration_s)
        except Exception:
            self.logger.warning("Per-unit burst feature extraction failed.", exc_info=True)
            return {}

        summary = result.get("summary", {})
        self.logger.info(
            "Per-unit bursts: %s/%s units bursting (MaxInterval), %s (logISI); "
            "%s units used an adaptive logISI threshold.",
            summary.get("n_bursting_units_maxinterval"), summary.get("n_units"),
            summary.get("n_bursting_units_logisi"),
            summary.get("n_units_logisi_adaptive_threshold"),
        )
        return result

    def _resolve_recording_metadata(self):
        """Return (recording, sampling_rate_hz, duration_s); entries may be None.

        `--reanalyze-bursts` together with `--skip-spikesorting` never
        populates self.recording, so fall back to opening the source file for
        metadata only — no preprocessing and no binary cache.
        """
        recording = self.recording
        if recording is None:
            try:
                recording = self._load_recording_file()
            except Exception:
                self.logger.warning(
                    "Could not load recording for fs/duration metadata; burst rates "
                    "will fall back to the spike-span duration.", exc_info=True,
                )
                return None, None, None

        try:
            fs = float(recording.get_sampling_frequency())
            duration_s = float(recording.get_num_frames()) / fs
        except Exception as e:
            self.logger.warning("Could not read fs/duration from recording: %s", e)
            return recording, None, None

        return recording, fs, duration_s

    def _run_burst_analysis(self, ids_list=None, plot_mode='separate', plot_debug=False,
                            raster_sort='none', fixed_y=False):
        self.logger.info("Running Network Burst Analysis...")

        spike_times = {}

        # 1. Load Spike Times
        if self.sorting:
            fs = self.recording.get_sampling_frequency()
            if ids_list is None:
                ids_list = self.analyzer.unit_ids

            missing_unit_ids = []
            for uid in ids_list:
                try:
                    spike_times[uid] = self.sorting.get_unit_spike_train(uid) / fs
                except KeyError:
                    missing_unit_ids.append(uid)

            if missing_unit_ids:
                self.logger.warning(
                    "Skipping %d unit(s) not present in active sorting during burst analysis: %s",
                    len(missing_unit_ids),
                    missing_unit_ids[:20],
                )

            if not spike_times:
                self.logger.error(
                    "No valid units left for burst analysis after filtering missing unit IDs."
                )
                return

            np.save(self.output_dir / "spike_times.npy", spike_times)
        else:
            spike_times_file = self.output_dir / "spike_times.npy"
            if spike_times_file.exists():
                try:
                    loaded = np.load(spike_times_file, allow_pickle=True).item()
                    if isinstance(loaded, dict):
                        if ids_list is not None:
                            id_set = {str(uid) for uid in ids_list}
                            spike_times = {
                                uid: st for uid, st in loaded.items()
                                if str(uid) in id_set
                            }
                        else:
                            spike_times = loaded
                        self.logger.info("Loaded existing spike times from %s", spike_times_file)
                except Exception as e:
                    self.logger.error("Failed loading spike times from %s: %s", spike_times_file, e)

            if not spike_times:
                self.logger.error("No spike times found for burst analysis.")
                return

        if not spike_times:
            self.logger.warning("Spike times dictionary is empty. Skipping burst analysis.")
            return

        try:
            # A. Run network burst detector
            detector_name = getattr(self, "burst_detector", "parameter_free")
            detector_fn = BURST_DETECTORS.get(detector_name)
            if detector_fn is None:
                self.logger.error(
                    "Unknown or unavailable burst detector '%s' (available: %s)",
                    detector_name, [k for k, v in BURST_DETECTORS.items() if v is not None],
                )
                return

            # Recording duration is the denominator for every rate the detector
            # reports, so it has to be resolved before detection rather than
            # bolted onto the JSON afterwards. Without it the detectors fall
            # back to the first-to-last-spike span and overstate the rates of
            # any well that goes quiet partway through.
            rec_for_meta, fs_meta, recording_duration_s = self._resolve_recording_metadata()

            detector_kwargs = {}
            if detector_name == "gaussian":
                detector_kwargs = dict(getattr(self, "gaussian_burst_kwargs", {}) or {})
            else:
                detector_kwargs = dict(getattr(self, "parameter_free_burst_kwargs", {}) or {})
            if recording_duration_s:
                detector_kwargs["duration_s"] = recording_duration_s

            network_data = detector_fn(SpikeTimes=spike_times, **detector_kwargs)

            if isinstance(network_data, dict) and "error" in network_data:
                self.logger.error(f"Burst detector returned error: {network_data['error']}")
                return

            # B. Extract array and tabular data before JSON serialization
            plot_data  = network_data.pop("plot_data", {})
            unit_stats = network_data.pop("unit_stats", {})

            # B2. What actually gets plotted vs. what gets saved to
            # network_plot_data.npz can differ: the gaussian detector only
            # has a population firing-rate signal (no participation-fraction
            # detection), so its plot should show only that — one line, no
            # twin axis, no participation trace — rather than reusing the
            # participation-fraction slot that parameter_free detectors fill.
            # network_plot_kwargs is what gets passed to plot_clean_network;
            # plot_data (unmodified) is still what's saved to the npz below.
            if detector_name == "gaussian":
                diagnostics = network_data.get("diagnostics", {})
                network_plot_kwargs = dict(plot_data)
                network_plot_kwargs["participation_fraction_signal"] = plot_data.get("population_firing_rate_hz")
                network_plot_kwargs["population_firing_rate_hz"] = None
                network_plot_kwargs["participation_baseline"] = diagnostics.get("baseline_mean_hz")
                network_plot_kwargs["detection_threshold"] = diagnostics.get("detection_threshold_hz")
                network_plot_kwargs["primary_ylabel"] = "Population Firing Rate (Hz)"
                network_plot_kwargs["primary_label"] = "Population firing rate"
            else:
                network_plot_kwargs = plot_data

            # C. Save plot_data as npz — large float arrays, not suited for JSON
            if plot_data:
                np.savez(
                    self.output_dir / "network_plot_data.npz",
                    **{k: np.asarray(v) for k, v in plot_data.items()}
                )
                self.logger.info("Saved network_plot_data.npz")

            # D. Per-unit burst features, then unit_stats.csv
            # Network bursts alone cannot separate "fewer neurons bursting"
            # from "each neuron bursting less"; these are the per-unit half of
            # the standard phenotyping feature set.
            unit_level = self._compute_unit_level_features(spike_times, recording_duration_s)
            per_unit_bursts = unit_level.pop("units", {}) if unit_level else {}

            df_units = self._build_unit_stats_frame(unit_stats, per_unit_bursts)
            if df_units is not None:
                df_units.to_csv(self.output_dir / "unit_stats.csv")
                self.logger.info("Saved unit_stats.csv (%d units, %d columns)",
                                 len(df_units), len(df_units.columns))

            # E. Save lean JSON
            network_data_clean = helper.recursive_clean(network_data)
            network_data_clean["n_units"] = len(spike_times)

            if fs_meta is not None:
                network_data_clean["fs"] = fs_meta
            if recording_duration_s is not None:
                network_data_clean["duration_s"] = recording_duration_s
            network_data_clean["detector"] = detector_name
            network_data_clean["detector_params"] = helper.recursive_clean(detector_kwargs)
            if unit_level:
                network_data_clean["unit_level"] = helper.recursive_clean(unit_level)
            network_data_clean["curation"] = getattr(self, "curation_summary", None)
            network_data_clean["provenance"] = collect_provenance()
            # Genotype/line/DIV etc. Without this block the results identify the
            # well but not the experiment, and no group comparison is possible.
            network_data_clean["sample"] = getattr(self, "sample", None)
            network_data_clean["project"] = self.project_name
            network_data_clean["date"] = str(self.date)
            network_data_clean["chip_id"] = self.chip_id
            network_data_clean["run_id"] = self.run_id
            network_data_clean["well"] = self.well

            temp_file = self.output_dir / "network_results.tmp.json"
            final_file = self.output_dir / "network_results.json"

            with open(temp_file, "w") as f:
                json.dump(network_data_clean, f, indent=2)

            if temp_file.exists():
                os.replace(temp_file, final_file)
                self.logger.info(f"Successfully saved: {final_file}")

            sorted_units = self._sort_units_for_raster(spike_times, raster_sort)

            ax_network_red = None

            _rc = {
                "font.size": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.direction": "out",
                "ytick.direction": "out",
            }

            with plt.rc_context(_rc):
                if plot_mode == "separate":
                    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                    ax_raster, ax_network = axs

                    helper.plot_clean_raster(
                        ax_raster, spike_times, sorted_units,
                        color="#2d3436", markersize=3, markeredgewidth=0.4, alpha=0.85
                    )
                    ax_network, ax_network_red = helper.plot_clean_network(
                        ax_network, **network_plot_kwargs, use_twinx=True
                    )

                elif plot_mode == "merged":
                    fig, ax_raster = plt.subplots(figsize=(14, 6))

                    helper.plot_clean_raster(
                        ax_raster, spike_times, sorted_units,
                        color="#2d3436", markersize=3, markeredgewidth=0.4, alpha=0.85
                    )

                    ax_network = ax_raster.twinx()
                    ax_network, ax_network_red = helper.plot_clean_network(
                        ax_network, **network_plot_kwargs, use_twinx=False
                    )

                    ax_raster.spines["right"].set_visible(False)
                    ax_network.spines["right"].set_visible(True)

                else:
                    self.logger.warning(f"Unknown plot mode: {plot_mode}")
                    return

                burstlet_events      = network_data["burst_fragments"]["events"]
                network_burst_events = network_data["network_bursts"]["events"]
                superburst_events    = network_data["superbursts"]["events"]

                helper.mark_burst_hierarchy(
                    ax_raster=ax_raster,
                    ax_network=ax_network,
                    burstlets=burstlet_events,
                    network_bursts=network_burst_events,
                    superbursts=superburst_events,
                    show_raster_spans=False,
                    show_burstlet_ticks=True,
                    show_network_ticks=True,
                    show_superburst_bars=True,
                    min_superburst_duration_s=2.5
                )

                hierarchy_handles = [
                    Line2D([0], [0], color="#636e72",  lw=1.2, label="Burstlet"),
                    Line2D([0], [0], color="#0984e3",  lw=2.0, label="Network burst"),
                    Line2D([0], [0], color="#6c5ce7",  lw=2.2, label="Superburst"),
                    Line2D([0], [0], marker='o', color='#d63031', lw=0, markersize=5, label="NB peak"),
                ]
                ax_network.legend(handles=hierarchy_handles, loc="upper right",
                                  frameon=False, ncol=2)

                plt.tight_layout()
                if plot_mode == "separate":
                    plt.subplots_adjust(hspace=0.10)

            full_svg   = self.output_dir / "raster_burst_plot.svg"
            full_png   = self.output_dir / "raster_burst_plot.png"
            zoom60_svg = self.output_dir / "raster_burst_plot_60s.svg"
            zoom60_png = self.output_dir / "raster_burst_plot_60s.png"
            zoom30_svg = self.output_dir / "raster_burst_plot_30s.svg"
            zoom30_png = self.output_dir / "raster_burst_plot_30s.png"

            plt.savefig(full_svg)
            plt.savefig(full_png, dpi=300)

            ax_raster.set_xlim(0, 60)
            ax_network.set_xlim(0, 60)
            if ax_network_red is not None and ax_network_red is not ax_network:
                ax_network_red.set_xlim(0, 60)
            plt.savefig(zoom60_svg)
            plt.savefig(zoom60_png, dpi=150)

            ax_raster.set_xlim(0, 30)
            ax_network.set_xlim(0, 30)
            if ax_network_red is not None and ax_network_red is not ax_network:
                ax_network_red.set_xlim(0, 30)
            ax_network.set_xlabel("Time (s)")
            plt.savefig(zoom30_svg)
            plt.savefig(zoom30_png, dpi=150)

            # Write this well's network y-max to a project-level summary so
            # --fixed-y can compute a global max across all wells in a later run.
            try:
                y_max = float(ax_network.get_ylim()[1])
                summary_file = self.output_root / self.project_name / f"{self.project_name}_y_max_summary.json"
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                summary = {}
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                summary.setdefault(str(self.date), {}).setdefault(str(self.chip_id), {})[str(self.run_id)] = y_max
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.info("Updated y-max summary: %s (y_max=%.4f)", summary_file, y_max)
            except Exception as e:
                self.logger.warning("Failed to update y-max summary: %s", e)

            plt.close(fig)

            self.logger.info("Burst analysis plots saved successfully.")

            if fixed_y:
                summary_file = self.output_root / self.project_name / f"{self.project_name}_y_max_summary.json"
                if not summary_file.exists():
                    self.logger.error(f"No y-max summary found at {summary_file}. Run without --fixed-y first.")
                else:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    all_maxima = [
                        v for date in summary.values()
                        for chip in date.values()
                        for v in chip.values()
                    ]
                    global_max = max(all_maxima)
                    self.logger.info(f"Applying fixed y-max: {global_max:.4f}")

                    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    ax_raster2, ax_network2 = axs2
                    helper.plot_clean_raster(ax_raster2, spike_times, color='gray',
                                             markersize=4, markeredgewidth=0.5, alpha=1.0)
                    helper.plot_clean_network(ax_network2, **network_plot_kwargs)
                    ax_network2.set_ylim(0, global_max)
                    plt.tight_layout()
                    plt.subplots_adjust(hspace=0.05)
                    for start, end in [(sb["start_time_s"], sb["end_time_s"]) for sb in superburst_events]:
                        ax_network2.axvspan(start, end, color='gray', alpha=0.3)
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot.svg")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot.png", dpi=300)
                    ax_raster2.set_xlim(0, 60)
                    ax_network2.set_xlim(0, 60)
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_60s.svg")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_60s.png", dpi=150)
                    ax_raster2.set_xlim(0, 30)
                    ax_network2.set_xlim(0, 30)
                    ax_network2.set_xlabel("Time (s)")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_30s.svg")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_30s.png", dpi=150)
                    plt.close(fig2)

        except Exception as e:
            self.logger.error(f"Burst analysis error: {e}")
            traceback.print_exc()
            raise e

    def _sort_units_for_raster(self, spike_times, raster_sort):
        """Returns ordered list of unit keys for raster y-axis."""
        if raster_sort == 'none':
            return None

        if raster_sort == 'firing_rate':
            return sorted(spike_times.keys(), key=lambda uid: len(spike_times[uid]))

        elif raster_sort == 'unit_id':
            return sorted(spike_times.keys())

        self.logger.warning(f"Unknown raster_sort: {raster_sort}. Falling back to none.")
        return None

    def _patch_phy_binary_path(self, phy_folder: Path):
        """Create a relative symlink in phy_output/ so phy finds the binary without
        depending on absolute paths. Patches params.py dat_path to the filename only."""
        binary_dir = self.output_dir / "binary"
        if not binary_dir.exists():
            self.logger.warning("phy export: binary/ not found, TraceView will be unavailable")
            return

        raw_files = sorted(binary_dir.glob("traces_cached_seg*.raw"))
        if not raw_files:
            self.logger.warning("phy export: no traces_cached_seg*.raw in binary/, TraceView will be unavailable")
            return

        params_file = phy_folder / "params.py"
        if not params_file.exists():
            return

        for raw_file in raw_files:
            link = phy_folder / raw_file.name
            if not link.exists():
                try:
                    link.symlink_to(Path("..") / "binary" / raw_file.name)
                except Exception as e:
                    self.logger.warning("phy export: could not symlink %s: %s", raw_file.name, e)

        # Patch dat_path in params.py to use just the filename (relative to phy_output/)
        # so phy resolves it against its working directory rather than the original abs path.
        try:
            text = params_file.read_text()
            # SpikeInterface writes one dat_path line per segment for multi-segment, or a single line
            text = re.sub(
                r"(dat_path\s*=\s*)['\"].*?['\"]",
                lambda m: m.group(1) + repr(raw_files[0].name),
                text,
            )
            params_file.write_text(text)
            self.logger.info("phy export: patched params.py to relative binary path (%s)", raw_files[0].name)
        except Exception as e:
            self.logger.warning("phy export: could not patch params.py: %s", e)
