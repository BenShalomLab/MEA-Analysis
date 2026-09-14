import json
import shutil

import numpy as np
import spikeinterface.full as si
import spikeinterface.preprocessing as spre

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage

# Seconds dropped from the end of every recording. Maxwell files can end on a
# partially written chunk whose trailing samples are not valid signal.
TRIM_TAIL_S = 1.0

# Local common median reference geometry, in microns.
# Inner radius excludes the channels that share the reference channel's own
# spike footprint (a Maxwell soma footprint spans roughly 50 um), outer radius
# bounds the neighbourhood used for the median.
# These must NOT be equal: reference='local' selects an annulus
# inner < d <= outer, so inner == outer selects nothing, every channel is left
# unreferenced, and the step becomes a silent no-op.
CMR_INNER_RADIUS_UM = 30.0
CMR_OUTER_RADIUS_UM = 200.0

# If fewer than this fraction of channels have at least one neighbour inside
# the annulus, local referencing is not meaningful for this electrode layout
# and global CMR is used instead.
CMR_MIN_COVERAGE_FRAC = 0.5


class PreprocessingMixin:
    """Phase 1: load recording file and run preprocessing pipeline."""

    def _local_reference_coverage(self, rec, inner_um, outer_um):
        """Fraction of channels with >=1 neighbour in the annulus, and the
        median neighbour count. Returns None when locations are unavailable.

        Guards against the failure mode where the annulus is empty (or nearly
        so) for a sparse Maxwell electrode selection: SpikeInterface leaves
        such channels unreferenced and only warns, so a misconfigured radius
        silently disables the whole referencing step.
        """
        try:
            locations = np.asarray(rec.get_channel_locations(), dtype=float)
        except Exception as e:
            self.logger.warning("Could not read channel locations for CMR check: %s", e)
            return None

        if locations.ndim != 2 or locations.shape[0] == 0:
            return None

        distances = np.linalg.norm(
            locations[:, None, :] - locations[None, :, :], axis=-1
        )
        neighbour_counts = ((distances > inner_um) & (distances <= outer_um)).sum(axis=1)
        return float(np.mean(neighbour_counts > 0)), float(np.median(neighbour_counts))

    def _apply_common_reference(self, rec):
        """Local median reference where the geometry supports it, else global."""
        coverage = self._local_reference_coverage(
            rec, CMR_INNER_RADIUS_UM, CMR_OUTER_RADIUS_UM
        )

        if coverage is None:
            self.logger.warning(
                "No channel locations available; using global CMR."
            )
            self.metadata['cmr_mode'] = 'global_no_locations'
            return spre.common_reference(rec, reference='global', operator='median')

        covered_frac, median_neighbours = coverage
        self.logger.info(
            "Local CMR annulus %.0f-%.0f um: %.0f%% of channels have a neighbour "
            "(median %d neighbours per channel).",
            CMR_INNER_RADIUS_UM, CMR_OUTER_RADIUS_UM,
            covered_frac * 100, int(median_neighbours),
        )

        if covered_frac < CMR_MIN_COVERAGE_FRAC:
            self.logger.warning(
                "Only %.0f%% of channels have a neighbour in the local CMR annulus "
                "(< %.0f%% required) — this electrode selection is too sparse for "
                "local referencing; using global CMR instead.",
                covered_frac * 100, CMR_MIN_COVERAGE_FRAC * 100,
            )
            self.metadata['cmr_mode'] = 'global_sparse_layout'
            self.metadata['cmr_coverage_frac'] = covered_frac
            return spre.common_reference(rec, reference='global', operator='median')

        try:
            referenced = spre.common_reference(
                rec,
                reference='local',
                operator='median',
                local_radius=(CMR_INNER_RADIUS_UM, CMR_OUTER_RADIUS_UM),
            )
        except Exception as e:
            self.logger.warning("Local CMR failed (%s); using global CMR.", e)
            self.metadata['cmr_mode'] = 'global_local_failed'
            self.metadata['cmr_error'] = str(e)
            return spre.common_reference(rec, reference='global', operator='median')

        self.metadata['cmr_mode'] = 'local'
        self.metadata['cmr_inner_radius_um'] = CMR_INNER_RADIUS_UM
        self.metadata['cmr_outer_radius_um'] = CMR_OUTER_RADIUS_UM
        self.metadata['cmr_coverage_frac'] = covered_frac
        self.metadata['cmr_median_neighbours'] = median_neighbours
        return referenced

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
            try:
                self.recording = si.load(binary_folder)
            except Exception as e:
                self.logger.debug("si.load failed on %s (%s); trying load_extractor.",
                                  binary_folder, e)
                self.recording = si.load_extractor(binary_folder)
            return
        self._save_checkpoint(ProcessingStage.PREPROCESSING)
        self.logger.info("--- [Phase 1] Preprocessing ---")

        rec = self._load_recording_file()

        fs = rec.get_sampling_frequency()
        self.metadata['fs'] = fs
        total_frames = int(rec.get_num_frames())

        # Drop the final TRIM_TAIL_S seconds. Previously this computed
        # floor(total_frames), which equals total_frames, so nothing was ever
        # removed despite the log message claiming otherwise.
        trim_frames = int(round(TRIM_TAIL_S * fs))
        end_frame = total_frames - trim_frames
        if trim_frames > 0 and end_frame > trim_frames:
            self.logger.info(
                "Trimming recording: %d -> %d frames (removed last %.1f s).",
                total_frames, end_frame, TRIM_TAIL_S,
            )
            rec = rec.frame_slice(start_frame=0, end_frame=end_frame)
        else:
            self.logger.warning(
                "Recording too short to trim %.1f s (%d frames at %.0f Hz); keeping all frames.",
                TRIM_TAIL_S, total_frames, fs,
            )

        self.metadata['trim_tail_s'] = TRIM_TAIL_S
        self.metadata['duration_s'] = rec.get_num_frames() / fs

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

        rec = self._apply_common_reference(rec)

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

        preprocessing_summary = {
            "bad_channel_ids": [str(c) for c in bad_channel_ids],
            "n_channels_input": int(n_total_channels),
            "n_channels_kept": int(self.recording.get_num_channels()),
            "bad_channel_fraction": (
                len(bad_channel_ids) / n_total_channels if n_total_channels else 0.0
            ),
            "trim_tail_s": TRIM_TAIL_S,
            "duration_s": self.metadata.get('duration_s'),
            "sampling_frequency_hz": fs,
            "highpass_hz": 300,
            "cmr_mode": self.metadata.get('cmr_mode'),
            "cmr_inner_radius_um": self.metadata.get('cmr_inner_radius_um'),
            "cmr_outer_radius_um": self.metadata.get('cmr_outer_radius_um'),
            "cmr_coverage_frac": self.metadata.get('cmr_coverage_frac'),
        }
        with open(self.output_dir / "bad_channels.json", "w", encoding="utf-8") as f:
            json.dump(preprocessing_summary, f, indent=2)

        self._save_checkpoint(
            ProcessingStage.PREPROCESSING_COMPLETE,
            preprocessing=preprocessing_summary,
        )
