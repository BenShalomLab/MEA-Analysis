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
        if self.checkpoint_file.exists() and not self.force_restart:
            with open(self.checkpoint_file, 'r') as f:
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
            'error': None,
        }

    def _save_checkpoint(self, stage, **kwargs):
        self.state['stage'] = stage.value
        self.state['checkpoint_schema_version'] = CHECKPOINT_SCHEMA_VERSION
        self.state['last_updated'] = str(datetime.now())
        self.state.update(kwargs)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        self.logger.info(f"Checkpoint Saved: {stage.name}")

    def should_skip(self):
        if self.state['stage'] == ProcessingStage.REPORTS_COMPLETE.value and not self.force_restart:
            self.logger.info("Pipeline already completed. Skipping.")
            return True
        return False
