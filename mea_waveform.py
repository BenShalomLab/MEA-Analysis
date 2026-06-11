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

    def _extract_rawsortedspikes(self, *, max_spikes_per_unit=200, ms_before=1.0, ms_after=2.0, seed=0):
        # ── 1. Load curated spike times ────────────────────────────────────────
        spike_times_file = self.output_dir / "spike_times.npy"
        self.logger.debug("spike_times.npy: %s  exists=%s", spike_times_file, spike_times_file.exists())
        if not spike_times_file.exists():
            self.logger.warning("Skipping raw mean template extraction: spike_times.npy not found.")
            return None
        saved_spike_times = np.load(spike_times_file, allow_pickle=True).item()
        curated_unit_ids = list(saved_spike_times.keys())
        self.logger.info(
            "Raw mean template extraction: %d curated units, window=%.1f+%.1f ms",
            len(curated_unit_ids), ms_before, ms_after,
        )

        # ── 2. Load templates for primary-channel lookup only ──────────────────
        # Prefer analyzer_output (correct sparse→global channel mapping).
        # Fall back to phy_output only if analyzer is absent.
        template_data = None
        channel_ids = None
        templates_ind = None   # phy sparse index only
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
                # get_data() returns dense (n_units, n_time, n_all_channels) with zeros
                # outside the sparsity mask — extremum_local_idx is a global channel index.
                template_data = np.asarray(templates_ext.get_data())
                channel_ids = np.asarray(self.analyzer.channel_ids)
                template_index_for_unit = {str(uid): i for i, uid in enumerate(self.analyzer.unit_ids)}
                self.logger.debug(
                    "Templates from analyzer: shape=%s  n_channels=%d",
                    template_data.shape, len(channel_ids),
                )

        if template_data is None:
            phy_folder = self.output_dir / "phy_output"
            templates_path       = phy_folder / "templates.npy"
            channel_map_path     = phy_folder / "channel_map.npy"
            templates_ind_path   = phy_folder / "templates_ind.npy"
            spike_templates_path = phy_folder / "spike_templates.npy"
            self.logger.debug(
                "phy_folder exists=%s  templates=%s  templates_ind=%s",
                phy_folder.exists(), templates_path.exists(), templates_ind_path.exists(),
            )
            if phy_folder.exists() and templates_path.exists() and spike_templates_path.exists():
                template_data = np.asarray(np.load(templates_path, allow_pickle=False))
                spike_templates_arr = np.asarray(
                    np.load(spike_templates_path, allow_pickle=False)
                ).reshape(-1)
                channel_ids = (
                    np.asarray(np.load(channel_map_path, allow_pickle=False)).reshape(-1)
                    if channel_map_path.exists()
                    else np.arange(template_data.shape[-1], dtype=np.int64)
                )
                if templates_ind_path.exists():
                    templates_ind = np.asarray(np.load(templates_ind_path, allow_pickle=False))
                else:
                    self.logger.warning(
                        "phy templates_ind.npy missing — sparse templates will give "
                        "unreliable channel selection."
                    )
                phy_unit_ids = [int(x) for x in np.unique(spike_templates_arr)]
                # unit_id == template row index only for unmerged Kilosort output.
                # After auto_merge or UnitMatch this mapping is wrong; prefer analyzer_output.
                template_index_for_unit = {str(uid): int(uid) for uid in phy_unit_ids}
                self.logger.warning(
                    "Using phy templates for channel lookup. "
                    "If units were merged after sorting, unit IDs may not match template rows — "
                    "results may be wrong. Re-run with analyzer_output present."
                )
                self.logger.debug(
                    "Templates from phy: shape=%s  templates_ind=%s",
                    template_data.shape,
                    templates_ind.shape if templates_ind is not None else "missing",
                )

        if template_data is None:
            self.logger.warning(
                "Skipping raw mean template extraction: no templates found "
                "(need analyzer_output or phy_output)."
            )
            return None

        # ── 3. Load raw recording and derive window parameters ─────────────────
        self.logger.debug("Loading raw recording: %s", self.file_path)
        raw_recording = self._load_recording_file()
        if channel_ids is None:
            channel_ids = np.asarray(raw_recording.get_channel_ids())
        fs = float(raw_recording.get_sampling_frequency())
        n_frames = int(raw_recording.get_num_frames())

        n_before = int(round(ms_before / 1000.0 * fs))
        n_after  = int(round(ms_after  / 1000.0 * fs))
        window_samples = n_before + n_after

        rng = np.random.default_rng(seed=seed)
        self.logger.debug(
            "Recording: fs=%.0f Hz  n_frames=%d  window=%d+%d=%d samples",
            fs, n_frames, n_before, n_after, window_samples,
        )

        # ── 4. Per-unit: resolve primary channel + sample spike centers ───────────
        unit_meta = {}   # uid → {channel_id_str, channel_id_raw, valid_centers}
        n_no_template = 0

        for uid_key in curated_unit_ids:
            template_idx = template_index_for_unit.get(str(uid_key)) if template_index_for_unit else None
            if template_idx is None or template_idx >= template_data.shape[0]:
                n_no_template += 1
                continue

            template = np.asarray(template_data[template_idx])  # (n_time, n_all_ch) dense
            if template.ndim != 2 or template.shape[1] == 0:
                n_no_template += 1
                continue

            # get_data() returns dense (zeros outside sparsity mask) so extremum_local_idx
            # is already a global channel index for the analyzer path.
            # For the phy path with templates_ind it is a local sparse index remapped below.
            extremum_local_idx = int(np.argmin(np.min(template, axis=0)))
            if templates_ind is not None:
                recording_ch_idx = int(templates_ind[template_idx, extremum_local_idx])
                extremum_channel_id = channel_ids[recording_ch_idx]
            else:
                extremum_channel_id = channel_ids[extremum_local_idx]

            # Round seconds → samples (truncation misaligns by up to 1 sample).
            spike_samples = np.round(
                np.asarray(saved_spike_times[uid_key], dtype=np.float64) * fs
            ).astype(np.int64)

            # Filter boundary spikes across ALL spikes first, then random-sample up to limit.
            all_valid = [
                int(c) for c in spike_samples
                if n_before <= int(c) < n_frames - n_after
            ]
            if len(all_valid) > max_spikes_per_unit:
                chosen = rng.choice(len(all_valid), size=max_spikes_per_unit, replace=False)
                selected_centers = sorted(int(all_valid[i]) for i in chosen)
            else:
                selected_centers = sorted(all_valid)

            unit_meta[uid_key] = {
                "channel_id":     str(extremum_channel_id),
                "channel_id_raw": extremum_channel_id,
                "valid_centers":  selected_centers,
            }

        if n_no_template:
            self.logger.warning(
                "%d curated units had no matching template and were skipped.", n_no_template
            )

        # ── 5. Single streaming pass — one sequential block read per block ─────
        # Builds a global time-sorted event list then streams through the recording
        # in BLOCK_SIZE chunks.  This replaces n_units × n_spikes individual seeks
        # (e.g. 300 × 200 = 60 000 calls) with ~ceil(n_frames / BLOCK_SIZE) calls,
        # which is critical for performance on network-mounted NAS storage.
        BLOCK_SIZE = 50_000   # 5 s at 10 kHz; ~60 MB/block for 300 channels × float32

        all_spike_events = []  # (center, uid, ch_id_str)
        for uid, meta in unit_meta.items():
            for center in meta["valid_centers"]:
                all_spike_events.append((center, uid, meta["channel_id"]))
        all_spike_events.sort(key=lambda x: x[0])

        snippets_by_unit: dict = defaultdict(list)
        n_events   = len(all_spike_events)
        event_idx  = 0
        n_blocks   = 0

        for block_start in range(0, n_frames, BLOCK_SIZE):
            if event_idx >= n_events:
                break
            block_end = block_start + BLOCK_SIZE

            # Collect all events whose center falls in [block_start, block_end).
            block_events = []
            i = event_idx
            while i < n_events and all_spike_events[i][0] < block_end:
                block_events.append(all_spike_events[i])
                i += 1
            event_idx = i

            if not block_events:
                continue

            # Extend the read range by the window margins so edge-of-block spikes
            # have their full window available.  Boundary spikes were pre-filtered
            # in pass 4, so clamping to [0, n_frames] is purely defensive.
            read_start = max(0, block_start - n_before)
            read_end   = min(n_frames, block_end + n_after)

            needed_channels = list({ch for _, _, ch in block_events})
            block_data = np.asarray(
                raw_recording.get_traces(
                    start_frame=read_start,
                    end_frame=read_end,
                    channel_ids=needed_channels,
                ),
                dtype=np.float32,
            )  # (read_end - read_start, len(needed_channels))
            ch_to_col = {ch: idx for idx, ch in enumerate(needed_channels)}
            n_blocks += 1

            for center, uid, ch_id in block_events:
                local_start = center - n_before - read_start
                col = ch_to_col[ch_id]
                snippet = block_data[local_start: local_start + window_samples, col]
                if snippet.shape[0] == window_samples:
                    snippets_by_unit[uid].append(snippet.copy())

        self.logger.info(
            "Streaming pass complete: %d block reads for %d spike events",
            n_blocks, n_events,
        )

        # ── 6. Compute per-unit mean templates ─────────────────────────────────
        extracted_units = {}
        for uid_key in unit_meta:
            meta     = unit_meta[uid_key]
            snippets = snippets_by_unit.get(uid_key, [])
            raw_mean_template = (
                np.mean(snippets, axis=0).astype(np.float32)
                if snippets
                else np.full(window_samples, np.nan, dtype=np.float32)
            )
            extracted_units[str(uid_key)] = {
                "unit_id":          uid_key,
                "primary_channel":  meta["channel_id_raw"],
                "raw_mean_template": raw_mean_template,
                "n_spikes_used":    int(len(snippets)),
                "window_samples":   int(window_samples),
                "ms_before":        float(ms_before),
                "ms_after":         float(ms_after),
            }

        # Units that had no template get a NaN entry so they are present in the output.
        for uid_key in curated_unit_ids:
            if str(uid_key) not in extracted_units:
                extracted_units[str(uid_key)] = {
                    "unit_id":          uid_key,
                    "primary_channel":  None,
                    "raw_mean_template": np.full(window_samples, np.nan, dtype=np.float32),
                    "n_spikes_used":    0,
                    "window_samples":   int(window_samples),
                    "ms_before":        float(ms_before),
                    "ms_after":         float(ms_after),
                }

        self.logger.info(
            "Raw mean templates: %d/%d units extracted (%d with spikes)",
            len(extracted_units),
            len(curated_unit_ids),
            sum(1 for v in extracted_units.values() if v["n_spikes_used"] > 0),
        )

        output_path = self.output_dir / "raw_mean_templates.npy"
        np.save(output_path, extracted_units, allow_pickle=True)
        self.state["raw_mean_templates_file"] = str(output_path)
        self.logger.info("Saved raw mean templates: %s", output_path)
        return output_path
