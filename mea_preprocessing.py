import json
import shutil
from math import floor

import spikeinterface.full as si
import spikeinterface.preprocessing as spre

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage


class PreprocessingMixin:
    """Phase 1: load recording file and run preprocessing pipeline."""

    def _load_recording_file(self):
        fpath = str(self.file_path)
        if fpath.endswith(".h5"):
            return si.read_maxwell(fpath, stream_id=self.stream_id, rec_name=self.recording_num)
        elif fpath.endswith(".nwb"):
            return si.read_nwb(fpath, load_if_exists=True)
        elif self.file_path.is_dir():
            return si.load_extractor(self.file_path)
        raise ValueError(f"Unknown format: {fpath}")

    def run_preprocessing(self):
        binary_folder = self.output_dir / "binary"

        if self.preprocessed_recording is not None:
            self.logger.info(
                "Using injected preprocessed recording (skip_preprocessing=%s)",
                bool(self.skip_preprocessing),
            )
            self.recording = self.preprocessed_recording
            if int(self.state.get('stage', 0)) < ProcessingStage.PREPROCESSING_COMPLETE.value:
                self._save_checkpoint(
                    ProcessingStage.PREPROCESSING_COMPLETE,
                    failed_stage=None,
                    error=None,
                    preprocessing_source="injected_preprocessed_recording",
                )
            return

        if self.state['stage'] >= ProcessingStage.PREPROCESSING_COMPLETE.value and binary_folder.exists():
            self.logger.info("Resuming: Loading preprocessed data from binary cache.")
            try: self.recording = si.load(binary_folder)
            except: self.recording = si.load_extractor(binary_folder)
            return
        self._save_checkpoint(ProcessingStage.PREPROCESSING)
        self.logger.info("--- [Phase 1] Preprocessing ---")

        rec = self._load_recording_file()

        fs = rec.get_sampling_frequency()
        self.metadata['fs'] = fs
        total_frames = rec.get_num_frames()
        end_frame = floor(total_frames)
        if end_frame > 0:
            self.logger.info(f"Trimming recording: {total_frames} -> {end_frame} frames (removed last 1s).")
            rec = rec.frame_slice(start_frame=0, end_frame=end_frame)

        if rec.get_dtype().kind == 'u':
            rec = spre.unsigned_to_signed(rec)

        rec = spre.highpass_filter(rec, freq_min=300)

        # --- Bad channel detection/removal ---
        # Faulty/dead/noisy electrodes left in the recording otherwise pass straight
        # through to spike detection (esp. the --skip-spikesorting path, which has no
        # sorter-side curation to catch them) and can register as physiologically
        # impossible firing rates (~1000+ Hz) that show up as solid horizontal bands
        # on raster plots.
        # method choice: 'coherence+psd' (and 'neighborhood_r2') score a channel against
        # its *spatial* neighbors, which assumes a dense, evenly-pitched probe (e.g.
        # Neuropixels, ~20-25 um pitch). Maxwell's sparse activity-scan electrode
        # selection has a median nearest-neighbor distance of 55-65 um and a chunk of
        # channels with no neighbor within 100 um at all — neighbor-based methods
        # misread that spatial sparsity as decorrelation and can flag the majority of
        # channels as bad on perfectly good recordings (verified: raw per-channel std
        # was normal and comparable across wells even when coherence+psd flagged
        # 74-98% of channels). 'mad' is amplitude-only (robust to occasional large
        # spikes, unlike plain 'std') and doesn't depend on neighbor geometry, so it's
        # the correct method for this array, not just a fallback.
        MAX_BAD_FRACTION = 0.3
        n_total_channels = rec.get_num_channels()

        def _try_detect(method):
            try:
                ids, _labels = spre.detect_bad_channels(rec, method=method)
                ids = list(ids) if ids is not None else []
                frac = (len(ids) / n_total_channels) if n_total_channels else 0.0
                if frac > MAX_BAD_FRACTION:
                    self.logger.warning(
                        "detect_bad_channels(%s) flagged %d/%d channels (%.0f%%) as bad — "
                        "implausible, treating as a failed detection rather than removing them.",
                        method, len(ids), n_total_channels, frac * 100,
                    )
                    return None
                return ids
            except Exception as e:
                self.logger.warning("detect_bad_channels(%s) failed (%s).", method, e)
                return None

        bad_channel_ids = _try_detect('mad')
        if bad_channel_ids is None:
            self.logger.warning("Falling back to 'std' method for bad-channel detection.")
            bad_channel_ids = _try_detect('std')
        if bad_channel_ids is None:
            self.logger.warning("Bad-channel detection failed entirely; no channels removed.")
            bad_channel_ids = []

        if bad_channel_ids:
            self.logger.warning(
                "Removing %d bad channel(s) detected during preprocessing: %s",
                len(bad_channel_ids), bad_channel_ids,
            )
            rec = rec.remove_channels(bad_channel_ids)
        else:
            self.logger.info("No bad channels detected during preprocessing.")

        self.metadata['bad_channel_ids'] = [str(c) for c in bad_channel_ids]

        # NOTE: local_radius=(250, 250) creates an annulus — inner radius 250 µm excluded.
        # If intent is all channels within 250 µm use (0, 250). Kept as-is to preserve
        # existing behaviour pending confirmation.
        try:
            rec = spre.common_reference(rec, reference='local', operator='median', local_radius=(250, 250))
        except:
            self.logger.warning("Local CMR failed (missing locations?), using Global CMR.")
            rec = spre.common_reference(rec, reference='global', operator='median')

        rec.annotate(is_filtered=True)

        if rec.get_dtype() != 'float32':
            self.logger.info("Converting to float32 to preserve signal fidelity...")
            rec = spre.astype(rec, 'float32')

        if binary_folder.exists():
            shutil.rmtree(binary_folder)

        self.logger.info(f"Saving binary recording to {binary_folder}...")
        rec.save(
            folder=binary_folder,
            format='binary',
            overwrite=True,
            n_jobs=(int(self.n_jobs) if self.n_jobs is not None else 16),
            chunk_duration=(str(self.chunk_duration) if self.chunk_duration is not None else '1s'),
            progress_bar=self.verbose
        )

        self.recording = si.load(binary_folder)

        with open(self.output_dir / "bad_channels.json", "w", encoding="utf-8") as f:
            json.dump({"bad_channel_ids": [str(c) for c in bad_channel_ids]}, f, indent=2)

        self._save_checkpoint(ProcessingStage.PREPROCESSING_COMPLETE)
