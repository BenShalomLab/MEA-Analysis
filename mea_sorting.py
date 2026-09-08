import json
import traceback
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import spikeinterface.full as si

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage


class SortingMixin:
    """Phase 2: spike sorting (Kilosort4) or spike detection only."""

    # Sanity-check thresholds applied after threshold-crossing-only spike detection
    # (--skip-spikesorting has no sorter-side curation, so bad channels that slip past
    # preprocessing's detect_bad_channels() still need a last-resort filter here).
    # 100 Hz is well above sustained firing rates seen even in fast-spiking interneurons
    # in MEA cultures; a channel averaging above this for the whole recording is almost
    # always continuous noise-threshold crossings, not real units.
    MAX_PHYSIOLOGICAL_HZ = 100.0
    REFRACTORY_VIOLATION_MS = 1.5
    REFRACTORY_VIOLATION_FRAC = 0.01

    def run_sorting(self):
        sorter_folder = self.output_dir / "sorter_output"
        if self.state['stage'] >= ProcessingStage.SORTING_COMPLETE.value:
            self.logger.info("Resuming: Loading existing sorting.")
            try: self.sorting = si.read_sorter_folder(sorter_folder)
            except: self.sorting = si.read_kilosort(sorter_folder)
            return
        self._save_checkpoint(ProcessingStage.SORTING)
        self.logger.info(f"--- [Phase 2] Spike Sorting ({self.sorter}) ---")

        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)} with {total_vram:.2f} GB VRAM")
        else:
            self.logger.warning("No GPU detected! Kilosort4 will likely fail or run extremely slowly on CPU.")
            total_vram = 0

        ks_params_high_vram = {
            'batch_size': int(self.recording.get_sampling_frequency()) * 2,
            'clear_cache': True,
            'invert_sign': True,
            'cluster_downsampling': 20,
            'max_cluster_subset': None,
            'nblocks': 0,
            'dmin': 17,
            'do_correction': False,
        }
        ks_params_low_vram = {
            'batch_size': int(self.recording.get_sampling_frequency() * 0.5),
            'clear_cache': True,
            'invert_sign': True,
            'cluster_downsampling': 30,
            'max_cluster_subset': 50000,
            'nblocks': 0,
            'do_correction': False,
        }

        ks_params = ks_params_high_vram if total_vram >= 14 else ks_params_low_vram

        if getattr(self, "sorter_kwargs", None):
            try:
                ks_params = dict(ks_params)
                ks_params.update(dict(self.sorter_kwargs))
            except Exception:
                pass

        start = timer()
        try:
            self.logger.info(f"Running Kilosort4 with parameters: {ks_params}")
            torch.cuda.empty_cache()
            self.sorting = si.run_sorter(
                sorter_name=self.sorter,
                recording=self.recording,
                folder=sorter_folder,
                delete_output_folder=False,
                remove_existing_folder=True,
                verbose=self.verbose,
                docker_image=self.docker_image,
                **ks_params
            )
            self.logger.info(f"Sorting finished in {timer()-start:.2f}s")
            self.logger.info("Cleaning sorting (removing excess spikes)...")
            self.sorting = si.remove_excess_spikes(self.sorting, self.recording)
            self.sorting = self.sorting.remove_empty_units()
            self._save_checkpoint(ProcessingStage.SORTING_COMPLETE, failed_stage=None, error=None)
        except Exception as e:
            err = {
                "failed_stage": ProcessingStage.SORTING.name,
                "exception": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "time": str(datetime.now())
            }
            self.logger.error(err["traceback"])
            self._save_checkpoint(ProcessingStage.PREPROCESSING_COMPLETE, error=err)
            raise

    def _spike_detection_only(self):
        """Detects spikes (threshold crossings) without sorting."""
        self.logger.info("--- [Phase 2-Alt] Spike Detection (No Sorting) ---")
        job_kwargs = {
            'n_jobs': (int(self.n_jobs) if self.n_jobs is not None else 16),
            'chunk_duration': (str(self.chunk_duration) if self.chunk_duration is not None else '1s'),
            'progress_bar': self.verbose,
        }

        peaks = detect_peaks(
            self.recording, method='by_channel',
            detect_threshold=5, peak_sign='neg', exclude_sweep_ms=0.1, **job_kwargs
        )

        self.logger.info(f"Detected {len(peaks)} total spikes.")
        fs = self.recording.get_sampling_frequency()
        self.metadata['fs'] = fs
        channel_ids = self.recording.get_channel_ids()
        duration_s = self.recording.get_num_frames() / fs
        spike_times = {}

        for ch_index, ch_id in enumerate(channel_ids):
            mask = peaks['channel_index'] == ch_index
            spike_times[ch_id] = (
                peaks['sample_index'][mask] / fs
                if np.any(mask)
                else np.array([])
            )

        spike_times, excluded = self._exclude_implausible_channels(spike_times, duration_s)

        np.save(self.output_dir / "spike_times.npy", spike_times)
        if excluded:
            with open(self.output_dir / "excluded_channels.json", "w", encoding="utf-8") as f:
                json.dump(excluded, f, indent=2)
        return list(spike_times.keys())

    def _exclude_implausible_channels(self, spike_times, duration_s):
        """Drop channels whose threshold-crossing rate/ISI pattern is not physiologically
        plausible (i.e. noise, not a real unit) before they reach burst analysis/plots."""
        kept = {}
        excluded = []
        for ch_id, times in spike_times.items():
            rate_hz = (len(times) / duration_s) if duration_s > 0 else 0.0
            isi_ms = np.diff(np.sort(times)) * 1000.0 if len(times) > 1 else np.array([])
            violation_frac = (
                float(np.mean(isi_ms < self.REFRACTORY_VIOLATION_MS)) if isi_ms.size else 0.0
            )

            if rate_hz > self.MAX_PHYSIOLOGICAL_HZ:
                excluded.append({
                    "channel_id": str(ch_id),
                    "firing_rate_hz": rate_hz,
                    "refractory_violation_frac": violation_frac,
                    "reason": "firing_rate_exceeds_physiological_max",
                })
                continue
            kept[ch_id] = times

        if excluded:
            self.logger.warning(
                "Excluding %d channel(s) with implausible firing rate (> %.0f Hz): %s",
                len(excluded), self.MAX_PHYSIOLOGICAL_HZ,
                [e["channel_id"] for e in excluded],
            )
        return kept, excluded
