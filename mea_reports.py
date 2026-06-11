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
    from parameter_free_burst_detector import compute_network_bursts
    import helper_functions as helper
    from scalebury import add_scalebar
except ImportError:
    try:
        from MEA_Analysis.IPNAnalysis.parameter_free_burst_detector import compute_network_bursts
        from MEA_Analysis.IPNAnalysis import helper_functions as helper
        from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar
    except ImportError:
        compute_network_bursts = None
        helper = None
        add_scalebar = None


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
            else:
                self.logger.info("Applying curation.")
                clean_metrics, rejection_log = self._apply_curation_logic(q_metrics, thresholds)
                clean_units = clean_metrics.index.values
                clean_metrics.to_excel(self.output_dir / "metrics_curated.xlsx")
                rejection_log.to_excel(self.output_dir / "rejection_log.xlsx")
                t_metrics.loc[clean_units].to_excel(self.output_dir / "tm_curated.xlsx")

            if len(clean_units) == 0:
                self.logger.warning("No units passed curation.")
                self._save_checkpoint(ProcessingStage.REPORTS_COMPLETE, n_units=0)
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
                                  failed_stage=None, error=None)
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
        defaults = {'presence_ratio': 0.75, 'rp_contamination': 0.15, 'firing_rate': 0.05,
                    'amplitude_median': -20, 'amplitude_cv_median': 0.5}
        if user_thresholds:
            defaults.update(user_thresholds)

        keep_mask = np.ones(len(metrics), dtype=bool)
        rejections = []
        for idx, row in metrics.iterrows():
            reasons = []
            if row.get('presence_ratio', 1) < defaults['presence_ratio']: reasons.append("Low Presence")
            if row.get('rp_contamination', 0) > defaults['rp_contamination']: reasons.append("High Contam")
            if row.get('firing_rate', 0) < defaults['firing_rate']: reasons.append("Low FR")
            if row.get('amplitude_median', -100) > defaults['amplitude_median']: reasons.append("Low Amp")
            #TODO: add cv_median logic after checking if metric exists in current version of SI

            if reasons:
                keep_mask[metrics.index.get_loc(row.name)] = False
                rejections.append({"unit_id": row.name, "reasons": "; ".join(reasons)})

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
            network_data = compute_network_bursts(SpikeTimes=spike_times)

            if isinstance(network_data, dict) and "error" in network_data:
                self.logger.error(f"Burst detector returned error: {network_data['error']}")
                return

            # B. Extract array and tabular data before JSON serialization
            plot_data  = network_data.pop("plot_data", {})
            unit_stats = network_data.pop("unit_stats", {})

            # C. Save plot_data as npz — large float arrays, not suited for JSON
            if plot_data:
                np.savez(
                    self.output_dir / "network_plot_data.npz",
                    **{k: np.asarray(v) for k, v in plot_data.items()}
                )
                self.logger.info("Saved network_plot_data.npz")

            # D. Save unit_stats as CSV
            if unit_stats:
                df_units = pd.DataFrame.from_dict(unit_stats, orient="index")
                df_units.index.name = "unit_id"
                df_units.to_csv(self.output_dir / "unit_stats.csv")
                self.logger.info("Saved unit_stats.csv")

            # E. Save lean JSON
            network_data_clean = helper.recursive_clean(network_data)
            network_data_clean["n_units"] = len(spike_times)

            temp_file = self.output_dir / "network_results.tmp.json"
            final_file = self.output_dir / "network_results.json"

            with open(temp_file, "w") as f:
                json.dump(network_data_clean, f, indent=2)

            if temp_file.exists():
                os.replace(temp_file, final_file)
                self.logger.info(f"Successfully saved: {final_file}")

            sorted_units = self._sort_units_for_raster(spike_times, raster_sort)

            ax_network_red = None

            if plot_mode == "separate":
                fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax_raster, ax_network = axs

                helper.plot_clean_raster(
                    ax_raster, spike_times, sorted_units,
                    color="gray", markersize=4, markeredgewidth=0.5, alpha=1.0
                )
                ax_network, ax_network_red = helper.plot_clean_network(
                    ax_network, **plot_data, use_twinx=True
                )

            elif plot_mode == "merged":
                fig, ax_raster = plt.subplots(figsize=(12, 5))

                helper.plot_clean_raster(
                    ax_raster, spike_times, sorted_units,
                    color="gray", markersize=4, markeredgewidth=0.5, alpha=1.0
                )

                ax_network = ax_raster.twinx()
                ax_network, ax_network_red = helper.plot_clean_network(
                    ax_network, **plot_data, use_twinx=False
                )

                ax_raster.spines["right"].set_visible(False)
                ax_network.spines["right"].set_visible(True)

            else:
                self.logger.warning(f"Unknown plot mode: {plot_mode}")
                return

            burstlet_events     = network_data["burstlets"]["events"]
            network_burst_events = network_data["network_bursts"]["events"]
            superburst_events   = network_data["superbursts"]["events"]

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
                Line2D([0], [0], color="black",       lw=1.2, label="Burstlet ticks"),
                Line2D([0], [0], color="steelblue",   lw=2.0, label="Network burst ticks"),
                Line2D([0], [0], color="mediumpurple", lw=2.2, label="Superbursts"),
                Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label="Network burst centers"),
            ]
            ax_raster.legend(handles=hierarchy_handles, loc="upper right", frameon=False, fontsize=8)

            plt.tight_layout()
            if plot_mode == "separate":
                plt.subplots_adjust(hspace=0.05)

            full_svg   = self.output_dir / "raster_burst_plot.svg"
            full_png   = self.output_dir / "raster_burst_plot.png"
            zoom60_svg = self.output_dir / "raster_burst_plot_60s.svg"
            zoom30_svg = self.output_dir / "raster_burst_plot_30s.svg"

            plt.savefig(full_svg)

            ax_raster.set_xlim(0, 60)
            ax_network.set_xlim(0, 60)
            if ax_network_red is not None and ax_network_red is not ax_network:
                ax_network_red.set_xlim(0, 60)
            plt.savefig(zoom60_svg)

            ax_raster.set_xlim(0, 30)
            ax_network.set_xlim(0, 30)
            if ax_network_red is not None and ax_network_red is not ax_network:
                ax_network_red.set_xlim(0, 30)
            ax_network.set_xlabel("Time (s)")
            plt.savefig(zoom30_svg)

            plt.savefig(full_png, dpi=300)

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
                    helper.plot_clean_network(ax_network2, **plot_data)
                    ax_network2.set_ylim(0, global_max)
                    plt.tight_layout()
                    plt.subplots_adjust(hspace=0.05)
                    for start, end in [(sb["start"], sb["end"]) for sb in superburst_events]:
                        ax_network2.axvspan(start, end, color='gray', alpha=0.3)
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot.svg")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot.png", dpi=300)
                    ax_raster2.set_xlim(0, 60)
                    ax_network2.set_xlim(0, 60)
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_60s.svg")
                    ax_raster2.set_xlim(0, 30)
                    ax_network2.set_xlim(0, 30)
                    ax_network2.set_xlabel("Time (s)")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_30s.svg")
                    plt.savefig(self.output_dir / "fixed_y_raster_burst_plot_30s.png", dpi=300)
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
