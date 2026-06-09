import shutil
import traceback
from datetime import datetime

import spikeinterface.full as si

try:
    from mea_checkpoint import ProcessingStage
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage


class AnalyzerMixin:
    """Phase 3: load existing sorting/analyzer and run the sorting analyzer."""

    def _load_existing_sorting(self):
        if self.sorting is not None:
            return True

        sorter_folder = self.output_dir / "sorting"
        if not sorter_folder.exists():
            self.logger.warning("Cannot load existing sorting: missing %s", sorter_folder)
            return False

        try:
            self.sorting = si.read_sorter_folder(sorter_folder)
        except Exception:
            try:
                self.sorting = si.read_kilosort(sorter_folder)
            except Exception as e:
                self.logger.warning("Failed loading existing sorting from %s: %s", sorter_folder, e)
                return False

        try:
            self.sorting = self.sorting.remove_empty_units()
        except Exception:
            pass
        return self.sorting is not None

    def _load_existing_analyzer(self):
        if self.analyzer is not None:
            return True

        analyzer_folder = self.output_dir / "analyzer_output"
        if not analyzer_folder.exists():
            self.logger.warning("Cannot load existing analyzer: missing %s", analyzer_folder)
            return False

        try:
            self.analyzer = si.load_sorting_analyzer(analyzer_folder)
            return True
        except Exception as e:
            self.logger.warning("Failed loading existing analyzer from %s: %s", analyzer_folder, e)
            return False

    def run_analyzer(self):
        analyzer_folder = self.output_dir / "analyzer_output"
        if (
            (not self.force_rerun_analyzer)
            and self.state['stage'] >= ProcessingStage.ANALYZER_COMPLETE.value
            and analyzer_folder.exists()
        ):
            self.logger.info("Resuming: Loading Sorting Analyzer.")
            self.analyzer = si.load_sorting_analyzer(analyzer_folder)
            self.sorting = self.analyzer.sorting
            return

        self._save_checkpoint(ProcessingStage.ANALYZER)
        self.logger.info("--- [Phase 3] Computing Sorting Analyzer ---")
        try:
            if analyzer_folder.exists():
                shutil.rmtree(analyzer_folder)

            sparsity = si.estimate_sparsity(
                self.sorting, self.recording,
                method="radius", radius_um=50, peak_sign='neg'
            )

            self.analyzer = si.create_sorting_analyzer(
                self.sorting,
                self.recording,
                format="binary_folder",
                folder=analyzer_folder,
                sparsity=sparsity,
                return_in_uV=True,
            )

            ext_list = ["random_spikes", "spike_amplitudes", "waveforms", "templates", "noise_levels",
                        "quality_metrics", "template_metrics", "unit_locations"]

            ext_params = {
                "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
                "unit_locations": {"method": "monopolar_triangulation"}
            }

            compute_kwargs = {'verbose': self.verbose}
            if self.n_jobs is not None:
                compute_kwargs['n_jobs'] = int(self.n_jobs)
            self.analyzer.compute(ext_list, extension_params=ext_params, **compute_kwargs)

            self.sorting = self.analyzer.sorting
            self._save_checkpoint(ProcessingStage.ANALYZER_COMPLETE, failed_stage=None, error=None)
        except Exception as e:
            err = {
                "failed_stage": ProcessingStage.ANALYZER.name,
                "exception": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "time": str(datetime.now())
            }
            self.logger.error(err["traceback"])
            try:
                fallback_stage = ProcessingStage(int(self.state.get('stage', ProcessingStage.MERGE_COMPLETE.value)))
            except Exception:
                fallback_stage = ProcessingStage.MERGE_COMPLETE
            self._save_checkpoint(fallback_stage, error=err)
            raise
