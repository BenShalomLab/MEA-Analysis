import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import spikeinterface.full as si

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage


class WaveformMixin:
    """Waveform extraction: processing info bookkeeping and raw mean template extraction."""

    def _write_processing_info(self, *, used_spike_sorting, reanalyze_bursts, extract_rawsortedspikes):
        payload = {
            "used_spike_sorting": bool(used_spike_sorting),
            "processing_mode": ("spike_sorting" if bool(used_spike_sorting) else "spike_detection_only"),
            "reanalyze_bursts": bool(reanalyze_bursts),
            "extract_rawsortedspikes": bool(extract_rawsortedspikes),
            "last_updated": str(datetime.now()),
        }
        info_file = self.output_dir / "processing_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.state.update(payload)

    def _extract_rawsortedspikes(self, *, max_spikes_per_unit=200, window_ms=2.5):
        # Flow:
        #  1. spike_times.npy  — curated units and their spike times (seconds)
        #  2. analyzer/phy templates — extremum channel lookup only
        #  3. raw recording    — extract and average waveforms per unit

        # ── 1. Load curated spike times ────────────────────────────────────────
        spike_times_file = self.output_dir / "spike_times.npy"
        self.logger.debug("spike_times.npy: %s  exists=%s", spike_times_file, spike_times_file.exists())
        if not spike_times_file.exists():
            self.logger.warning("Skipping raw mean template extraction: spike_times.npy not found.")
            return None
        saved_spike_times = np.load(spike_times_file, allow_pickle=True).item()
        curated_unit_ids = list(saved_spike_times.keys())
        self.logger.info("Raw mean template extraction: %d curated units, window=%.1f ms",
                         len(curated_unit_ids), window_ms)

        # ── 2. Load templates for extremum channel lookup ──────────────────────
        # Prefer analyzer (full-channel templates). Fall back to phy+templates_ind.
        template_data = None
        channel_ids = None
        templates_ind = None
        template_index_for_unit = None

        analyzer_folder = self.output_dir / "analyzer_output"
        self.logger.debug("analyzer_folder exists=%s", analyzer_folder.exists())
        if analyzer_folder.exists():
            self.analyzer = si.load_sorting_analyzer(analyzer_folder)
            templates_ext = self.analyzer.get_extension("templates")
            if templates_ext is None:
                self.analyzer.compute("templates", verbose=self.verbose)
                templates_ext = self.analyzer.get_extension("templates")
            if templates_ext is not None:
                template_data = np.asarray(templates_ext.get_data())
                analyzer_channel_ids = getattr(self.analyzer, "channel_ids", None)
                channel_ids = np.asarray(analyzer_channel_ids) if analyzer_channel_ids is not None else None
                template_index_for_unit = {str(uid): i for i, uid in enumerate(self.analyzer.unit_ids)}
                self.logger.debug("Templates from analyzer: shape=%s  channel_ids=%s",
                                  template_data.shape,
                                  channel_ids.shape if channel_ids is not None else "None")

        if template_data is None:
            phy_folder = self.output_dir / "phy_output"
            templates_path     = phy_folder / "templates.npy"
            channel_map_path   = phy_folder / "channel_map.npy"
            templates_ind_path = phy_folder / "templates_ind.npy"
            spike_templates_path = phy_folder / "spike_templates.npy"
            self.logger.debug("phy_folder exists=%s  templates=%s  templates_ind=%s",
                              phy_folder.exists(), templates_path.exists(), templates_ind_path.exists())
            if phy_folder.exists() and templates_path.exists() and spike_templates_path.exists():
                template_data = np.asarray(np.load(templates_path, allow_pickle=False))
                spike_templates_arr = np.asarray(np.load(spike_templates_path, allow_pickle=False)).reshape(-1)
                channel_ids = (
                    np.asarray(np.load(channel_map_path, allow_pickle=False)).reshape(-1)
                    if channel_map_path.exists()
                    else np.arange(template_data.shape[-1], dtype=np.int64)
                )
                if templates_ind_path.exists():
                    templates_ind = np.asarray(np.load(templates_ind_path, allow_pickle=False))
                else:
                    self.logger.warning(
                        "phy templates_ind.npy missing — sparse templates (%d ch/unit) will give "
                        "unreliable channel selection.", template_data.shape[-1]
                    )
                phy_unit_ids = [int(x) for x in np.unique(spike_templates_arr)]
                template_index_for_unit = {str(uid): int(uid) for uid in phy_unit_ids}
                self.logger.debug("Templates from phy: shape=%s  templates_ind=%s",
                                  template_data.shape,
                                  templates_ind.shape if templates_ind is not None else "missing")

        if template_data is None:
            self.logger.warning("Skipping raw mean template extraction: no templates found "
                                "(need analyzer_output or phy_output).")
            return None

        # ── 3. Load raw recording ──────────────────────────────────────────────
        self.logger.debug("Loading raw recording: %s", self.file_path)
        raw_recording = self._load_recording_file()
        if channel_ids is None:
            channel_ids = np.asarray(raw_recording.get_channel_ids())
        fs = float(raw_recording.get_sampling_frequency())
        n_frames = int(raw_recording.get_num_frames())
        window_samples = max(1, int(round((float(window_ms) / 1000.0) * fs)))
        if window_samples % 2 == 0:
            window_samples += 1
        half_window = window_samples // 2
        self.logger.debug("Recording: fs=%.0f Hz, n_frames=%d, window_samples=%d",
                          fs, n_frames, window_samples)

        # ── Pass 1: resolve extremum channel + spike centers per curated unit ──
        unit_meta = {}
        n_no_template = 0
        for uid_key in curated_unit_ids:
            template_idx = template_index_for_unit.get(str(uid_key)) if template_index_for_unit else None
            if template_idx is None or template_idx >= template_data.shape[0]:
                n_no_template += 1
                continue

            template = np.asarray(template_data[template_idx])  # [n_time, n_channels]
            if template.ndim != 2 or template.shape[1] == 0:
                n_no_template += 1
                continue

            channel_min_peaks = np.min(template, axis=0)
            extremum_local_idx = int(np.argmin(channel_min_peaks))
            if templates_ind is not None:
                recording_ch_idx = int(templates_ind[template_idx, extremum_local_idx])
                extremum_channel_id = channel_ids[recording_ch_idx]
            else:
                extremum_channel_id = channel_ids[extremum_local_idx]

            spike_samples = (np.asarray(saved_spike_times[uid_key], dtype=np.float64) * fs).astype(np.int64)
            valid_centers = [
                int(c) for c in spike_samples[: int(max_spikes_per_unit)]
                if int(c) - half_window >= 0 and int(c) - half_window + window_samples <= n_frames
            ]
            unit_meta[uid_key] = {
                "channel_id": str(extremum_channel_id),
                "channel_id_raw": extremum_channel_id,
                "valid_centers": valid_centers,
            }

        if n_no_template:
            self.logger.warning("%d curated units had no matching template and were skipped.", n_no_template)
        self.logger.debug("%d units have channel assignments", len(unit_meta))

        # --- Pass 2: group by channel, one get_traces() call per channel ---
        channel_to_units = defaultdict(list)
        for unit_id, meta in unit_meta.items():
            channel_to_units[meta["channel_id"]].append(unit_id)

        n_channels = len(channel_to_units)
        self.logger.info("Reading raw traces: %d unique channels for %d units",
                         n_channels, len(unit_meta))

        extracted_units = {}
        for ch_i, (ch_id, ch_unit_ids) in enumerate(channel_to_units.items()):
            all_centers = sorted({
                c for uid in ch_unit_ids for c in unit_meta[uid]["valid_centers"]
            })
            if not all_centers:
                for unit_id in ch_unit_ids:
                    extracted_units[str(unit_id)] = {
                        "unit_id": int(unit_id) if isinstance(unit_id, np.integer) else unit_id,
                        "primary_channel": unit_meta[unit_id]["channel_id_raw"],
                        "raw_mean_template": np.full(window_samples, np.nan, dtype=np.float32),
                        "n_spikes_used": 0,
                        "window_samples": int(window_samples),
                    }
                continue

            ch_start = all_centers[0] - half_window
            ch_end = all_centers[-1] - half_window + window_samples
            self.logger.debug(
                "Channel %s (%d/%d): %d units, reading frames [%d, %d] (%.1f s)",
                ch_id, ch_i + 1, n_channels, len(ch_unit_ids),
                ch_start, ch_end, (ch_end - ch_start) / fs,
            )
            ch_trace = np.asarray(
                raw_recording.get_traces(
                    start_frame=ch_start,
                    end_frame=ch_end,
                    channel_ids=[ch_id],
                ),
                dtype=np.float32,
            ).reshape(-1)

            for unit_id in ch_unit_ids:
                valid_centers = unit_meta[unit_id]["valid_centers"]
                snippets = []
                for center in valid_centers:
                    s = center - half_window - ch_start
                    snippet = ch_trace[s : s + window_samples]
                    if snippet.shape[0] == window_samples:
                        snippets.append(snippet)
                raw_mean_template = (
                    np.mean(snippets, axis=0).astype(np.float32)
                    if snippets
                    else np.full(window_samples, np.nan, dtype=np.float32)
                )
                ch_id_raw = unit_meta[unit_id]["channel_id_raw"]
                extracted_units[str(unit_id)] = {
                    "unit_id": int(unit_id) if isinstance(unit_id, np.integer) else unit_id,
                    "primary_channel": int(ch_id_raw) if isinstance(ch_id_raw, np.integer) else ch_id_raw,
                    "raw_mean_template": raw_mean_template,
                    "n_spikes_used": int(len(snippets)),
                    "window_samples": int(window_samples),
                }

        output_path = self.output_dir / "raw_mean_templates.npy"
        np.save(output_path, extracted_units, allow_pickle=True)
        self.state["raw_mean_templates_file"] = str(output_path)
        self.logger.info("Saved raw mean templates: %s", output_path)
        return output_path
