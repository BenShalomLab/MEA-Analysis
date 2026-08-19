import os
import re
import sys
import json
import logging
import configparser
from datetime import datetime
from pathlib import Path

try:
    from mea_checkpoint import ProcessingStage, CHECKPOINT_SCHEMA_VERSION
except ImportError:
    from MEA_Analysis.IPNAnalysis.mea_checkpoint import ProcessingStage, CHECKPOINT_SCHEMA_VERSION


def resolve_hdf5_plugin_path():
    """Set HDF5_PLUGIN_PATH from a known install location if it isn't set already.

    Must run before spikeinterface/h5py is imported anywhere in the process —
    the underlying HDF5 C library reads this env var once, at library init,
    and caches the plugin search path. Setting it from Python afterward (e.g.
    from inside MEAPipeline.__init__) has no effect on an already-initialized
    library, which is why this is a standalone function callable at module
    import time, not just an InfraMixin method.

    Docker images and sbatch.sh export HDF5_PLUGIN_PATH explicitly; a bare
    `salloc` + `conda activate` shell on NERSC doesn't, so h5py falls back to
    its compiled-in default plugin dir (doesn't exist there) and Maxwell .h5
    reads crash.

    Returns the path it set, or None if already set / nothing found.
    """
    if os.environ.get("HDF5_PLUGIN_PATH"):
        return None
    candidates = [
        Path.home() / "hdf5_plugin_path_maxwell",   # NERSC home-dir install
        Path("/opt/hdf5/plugins"),                  # baked into the Docker images
    ]
    if os.environ.get("SCRATCH"):
        candidates.append(Path(os.environ["SCRATCH"]) / "hdf5_plugin")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("libcompression.*")):
            os.environ["HDF5_PLUGIN_PATH"] = str(candidate)
            return candidate
    return None


class InfraMixin:
    """Infrastructure methods: logging, metadata parsing, checkpointing, runtime controls."""

    def _setup_logger(self, log_file):
        logger = logging.getLogger(f"mea_{self.stream_id}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            fh = logging.FileHandler(log_file, mode='a')
            fh.stream.write("\n" + "="*80 + "\n")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _apply_runtime_controls(self):
        if self.cuda_visible_devices is not None:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            except Exception:
                pass
        self._apply_hdf5_plugin_path_fallback()

    def _apply_hdf5_plugin_path_fallback(self):
        # The real work already happened at module-import time, before
        # spikeinterface/h5py was imported (see resolve_hdf5_plugin_path()
        # docstring for why it has to run that early). This just logs what
        # was resolved, now that self.logger exists — or re-checks in case
        # something cleared the env var between then and now.
        if os.environ.get("HDF5_PLUGIN_PATH"):
            self.logger.debug("HDF5_PLUGIN_PATH=%s", os.environ["HDF5_PLUGIN_PATH"])
            return
        candidate = resolve_hdf5_plugin_path()
        if candidate:
            self.logger.info("HDF5_PLUGIN_PATH not set — falling back to %s", candidate)
        else:
            self.logger.warning(
                "HDF5_PLUGIN_PATH not set and no Maxwell compression plugin found in "
                "known locations — raw .h5 reads will fail if the file is compressed."
            )

    def _log_runtime_controls(self):
        def _env_or_none(name):
            value = os.environ.get(name)
            if value is None:
                return None
            token = str(value).strip()
            return token if token else None

        self.logger.info(
            "Runtime snapshot: pid=%s cpu_count=%s n_jobs=%s chunk_duration=%s",
            os.getpid(),
            os.cpu_count(),
            self.n_jobs,
            self.chunk_duration,
        )
        self.logger.info(
            "Runtime controls: cuda_visible_devices=%s",
            self.cuda_visible_devices,
        )
        self.logger.info(
            "Runtime env effective: CUDA_VISIBLE_DEVICES=%s",
            _env_or_none("CUDA_VISIBLE_DEVICES"),
        )

    def _parse_metadata(self):
        meta = {
            'run_id': None, 'chip_id': None, 'project': None,
            'relative_pattern': f"{self.file_path.parent.parent.name}/{self.file_path.parent.name}/{self.file_path.name}",
            'date': None, 'well': None
        }

        # Strategy A: Regex on path (Fallback)
        try:
            path_str = str(self.file_path)
            match = re.search(r"/(\d+)/data.raw.h5", path_str)
            if match: meta['run_id'] = match.group(1)
            parts = path_str.split(os.sep)
            if len(parts) > 5:
                meta['relative_pattern'] = os.path.join(*parts[-6:-1])
                meta['project'] = parts[-6]
                meta['date'] = parts[-5]
                meta['chip_id'] = parts[-4]
                meta['well'] = self.stream_id
        except Exception: pass

        # Strategy B: .metadata file (Overrides regex)
        meta_file = self.file_path.parent / ".metadata"
        if meta_file.exists():
            try:
                cfg = configparser.ConfigParser()
                cfg.read(meta_file, encoding='utf-8')
                if 'properties' in cfg:
                    meta['run_id'] = cfg['properties'].get('runid', meta.get('run_id'))
                    meta['project'] = cfg['properties'].get('project_title', meta.get('project'))
                if 'runtime' in cfg:
                    meta['chip_id'] = cfg['runtime'].get('chipid', meta.get('chip_id'))
            except: pass
        return meta

    def _validate_output_subdir_after_well(self, value):
        if value is None:
            return None

        token = str(value).strip()
        if not token:
            return None

        if "/" in token or "\\" in token:
            raise ValueError(
                "output_subdir_after_well must be a single directory name, not a path"
            )

        candidate = Path(token)
        if candidate.is_absolute() or token in (".", ".."):
            raise ValueError(
                "output_subdir_after_well must be a relative single directory name"
            )

        return token

    def _load_checkpoint(self):
        # Prefer the primary (well-local) file; fall back to extra_checkpoint_file so
        # that runs started before this change (which only wrote the separate dir) resume cleanly.
        source = self.checkpoint_file
        if not source.exists() and getattr(self, 'extra_checkpoint_file', None) and self.extra_checkpoint_file.exists():
            source = self.extra_checkpoint_file
        if source.exists() and not self.force_restart:
            with open(source, 'r') as f:
                state = json.load(f)

            try:
                schema_version = int(state.get("checkpoint_schema_version", 1))
            except Exception:
                schema_version = 1
            if schema_version < CHECKPOINT_SCHEMA_VERSION:
                try:
                    old_stage = int(state.get("stage", ProcessingStage.NOT_STARTED.value))
                except Exception:
                    old_stage = ProcessingStage.NOT_STARTED.value
                if old_stage >= 5:
                    state["stage"] = old_stage + 2
                state["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION

            return state
        return {
            'stage': ProcessingStage.NOT_STARTED.value,
            'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
            'failed_stage': None,
            'last_updated': None,
            'run_id': self.run_id,
            'chip_id': self.chip_id,
            'well': self.well,
            'project': self.project_name,
            'date': self.date,
            'output_dir': str(self.output_dir),
            'data_dir': str(self.file_path),
            'error': None,
        }

    def _save_checkpoint(self, stage, **kwargs):
        self.state['stage'] = stage.value
        self.state['checkpoint_schema_version'] = CHECKPOINT_SCHEMA_VERSION
        self.state['last_updated'] = str(datetime.now())
        self.state.update(kwargs)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        if getattr(self, 'extra_checkpoint_file', None):
            try:
                with open(self.extra_checkpoint_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
            except Exception as e:
                self.logger.warning("Could not write extra checkpoint copy to %s: %s",
                                    self.extra_checkpoint_file, e)
        self.logger.info(f"Checkpoint Saved: {stage.name}")

    def should_skip(self):
        if self.state['stage'] == ProcessingStage.REPORTS_COMPLETE.value and not self.force_restart:
            self.logger.info("Pipeline already completed. Skipping.")
            return True
        return False
