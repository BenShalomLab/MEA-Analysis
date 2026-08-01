#!/usr/bin/env python3
"""
MEA recording watcher — detects when a run folder (e.g. ``000041``) has been
**completely dumped** to the watched directory, then triggers
``run_pipeline_driver.py`` on it with options supplied by the UI.

Design constraints
------------------
* **Read-only on the data server.** The watcher only *reads* run folders
  (file sizes, mtimes, and the MaxWell ``mxassay.metadata`` marker). It never
  writes anything into the watched directory.
* **The analysis repo is never modified.** The pipeline is invoked as a normal
  subprocess exactly as a human would run it from the command line.
* **All options come from the UI.** Input path, output path, and every
  ``run_pipeline_driver.py`` flag are supplied as a job config (see
  ``driver_schema.py``), so the UI is the single control surface.

Completion detection
--------------------
A run folder is dispatched only when all of these hold:

1. It contains a recording file (``data.raw.h5`` by default).
2. MaxWell wrote a ``finished=`` marker into ``mxassay.metadata``
   (optional — disable with ``require_finished_marker: false``).
3. **Quiescence:** nothing anywhere under the folder changed in size, mtime, or
   file count across two consecutive checks separated by ``settle_seconds``.
   This is what guarantees a multi-GB ``data.raw.h5`` has finished copying.

Detection is *stateless across polls*: the fingerprint from the previous scan is
remembered, so the watcher does not block while waiting for a folder to settle
and can supervise many runs concurrently.

Usage
-----
    # Driven by a job config written by the UI
    python orchestration/watcher.py --job-config /var/lib/mea-watcher/job.json

    # Ad-hoc from the command line
    python orchestration/watcher.py --watch-dir /mnt/server2/incoming \
        --output-dir /data/AnalyzedData --once --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver_schema import build_driver_args, default_options  # noqa: E402

LOG = logging.getLogger("mea.watcher")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRIVER = REPO_ROOT / "run_pipeline_driver.py"
ACTIVITY_SCRIPT = Path(__file__).resolve().parent / "activity_scan.py"

# The two independent analyses.
JOB_NETWORK = "network"
JOB_ACTIVITY = "activity"
JOB_LABELS = {JOB_NETWORK: "Network", JOB_ACTIVITY: "Activity scan"}
# State/logs live outside the watched (read-only) data server and outside the repo.
DEFAULT_WORK_DIR = Path(os.environ.get("MEA_WATCHER_HOME", Path.home() / ".mea-watcher"))


# --------------------------------------------------------------------------- #
# Job configuration (produced by the UI)
# --------------------------------------------------------------------------- #
@dataclass
class JobConfig:
    """Everything the UI supplies for a watch job."""

    watch_dir: str = ""                     # input path on the data server (read-only)
    driver_options: dict = field(default_factory=default_options)  # all driver flags

    # --- Analyses -----------------------------------------------------------
    # The two analyses are independent jobs: they are enabled separately, run as
    # separate processes, write to separate outputs, and are tracked separately.
    #
    # 1. Network  — run_pipeline_driver.py (spike sorting; needs a GPU)
    # 2. Activity — activity_scan.py (whole-array maps; CPU only, seconds)
    run_network: bool = True
    run_activity: bool = False

    # Detection tuning
    h5_glob: str = "data.raw.h5"
    # Only recordings under this path component are processed, matching the
    # driver's own filter. "Network" excludes ActivityScan. Blank = no filter.
    assay_subfolder: str = "Network"
    # Assay folder holding the activity scans.
    activity_subfolder: str = "ActivityScan"
    # Where activity-scan results go. Blank -> <output_dir>/ActivityScan.
    activity_output_dir: str = ""
    activity_active_hz: float = 0.05
    activity_figures: bool = True

    settle_seconds: int = 600
    poll_seconds: int = 30
    require_finished_marker: bool = True

    # Execution
    driver: str = str(DEFAULT_DRIVER)
    python: str = sys.executable
    work_dir: str = str(DEFAULT_WORK_DIR)   # where state + logs are written
    dry_run: bool = False

    @property
    def output_dir(self) -> Optional[str]:
        return self.driver_options.get("output_dir")

    @property
    def activity_out(self) -> Optional[str]:
        """Resolved output directory for activity-scan results."""
        if self.activity_output_dir:
            return self.activity_output_dir
        base = self.output_dir
        return str(Path(base) / "ActivityScan") if base else None

    def enabled_jobs(self) -> list[str]:
        jobs = []
        if self.run_network:
            jobs.append(JOB_NETWORK)
        if self.run_activity:
            jobs.append(JOB_ACTIVITY)
        return jobs

    def subfolder_for(self, job: str) -> str:
        return self.assay_subfolder if job == JOB_NETWORK else self.activity_subfolder

    @classmethod
    def load(cls, path: Path) -> "JobConfig":
        data = json.loads(Path(path).read_text())
        opts = default_options()
        opts.update(data.get("driver_options") or {})
        data["driver_options"] = opts
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.watch_dir:
            errs.append("Input path is required.")
        elif not Path(self.watch_dir).is_dir():
            errs.append(f"Input path does not exist or is not a directory: {self.watch_dir}")
        if not self.enabled_jobs():
            errs.append("Enable at least one analysis (Network or Activity scan).")
        if self.run_network and not Path(self.driver).exists():
            errs.append(f"run_pipeline_driver.py not found at: {self.driver}")
        if self.run_activity and not ACTIVITY_SCRIPT.exists():
            errs.append(f"activity_scan.py not found at: {ACTIVITY_SCRIPT}")
        out = self.output_dir
        if out:
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errs.append(f"Output path is not writable: {out} ({exc})")
        act = self.activity_out if self.run_activity else None
        if act:
            try:
                Path(act).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errs.append(f"Activity scan output path is not writable: {act} ({exc})")
        if self.settle_seconds < 1:
            errs.append("settle_seconds must be >= 1")
        if self.poll_seconds < 1:
            errs.append("poll_seconds must be >= 1")
        return errs


# --------------------------------------------------------------------------- #
# State store (idempotency + UI status feed)
# --------------------------------------------------------------------------- #
class StateStore:
    """Tracks each run's lifecycle. Written to the watcher host, never to the data server."""

    TERMINAL = {"done", "failed"}
    # A run in any of these states must never be dispatched again.
    # "detected" is the dry-run outcome and counts as claimed.
    CLAIMED = {"detected", "dispatched", "running", "done", "failed"}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                LOG.warning("Unreadable state file %s (%s); starting fresh", self.path, exc)

    # Every accessor takes the lock: the scan loop and each job thread write
    # while the API reads for /api/status, so unsynchronised reads could observe
    # a half-updated entry or race the dict against a concurrent write.
    def get(self, key: str) -> dict:
        with self._lock:
            return dict(self._data.get(key, {}))

    def status(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data.get(key)
            return entry.get("status") if entry else None

    def is_claimed(self, key: str) -> bool:
        return self.status(key) in self.CLAIMED

    def update(self, key: str, **fields) -> None:
        with self._lock:
            entry = self._data.setdefault(key, {"run": Path(key).name})
            entry.update(fields)
            self._flush()

    def all(self) -> dict[str, dict]:
        """Deep-enough copy for a consistent snapshot while jobs are running."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def reset(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Completion detection (read-only)
# --------------------------------------------------------------------------- #
def folder_fingerprint(folder: Path) -> tuple[int, int, float]:
    """(file_count, total_bytes, max_mtime) over the whole tree.

    An identical fingerprint at two points in time means the copy is finished.
    Returns a sentinel that never compares equal if a file vanishes mid-scan.
    """
    count = 0
    total = 0
    newest = 0.0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                st = (Path(root) / name).stat()
            except OSError:
                return (-1, -1, time.time())
            count += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return (count, total, newest)


def _metadata_says_finished(meta: Path) -> bool:
    """True if this mxassay.metadata records ``finished=`` under ``[runtime]``."""
    try:
        text = meta.read_text(errors="ignore")
    except OSError:
        return False
    in_runtime = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_runtime = line.lower() == "[runtime]"
        elif in_runtime and line.startswith("finished=") and len(line) > len("finished="):
            return True
    return False


def has_finished_marker(run_dir: Path, h5_glob: str = "data.raw.h5",
                        assay_subfolder: str = "Network") -> bool:
    """Whether every recording in this folder has been marked complete by MaxWell.

    MaxWell writes ``mxassay.metadata`` **next to each recording**, so in a
    nested layout (``<chip>/Network/<run_id>/data.raw.h5``) the marker lives in
    the run folder, not at the top of the dispatched folder. Check the sibling
    metadata of every qualifying recording, and require all of them.

    Falls back to a metadata file at the folder root for flat layouts.
    """
    recordings = find_recordings(run_dir, h5_glob, assay_subfolder)
    if not recordings:
        recordings = [p for p in run_dir.rglob(h5_glob)]

    metas = [r.parent / "mxassay.metadata" for r in recordings]
    metas = [m for m in metas if m.exists()]

    if metas:
        return all(_metadata_says_finished(m) for m in metas)

    root_meta = run_dir / "mxassay.metadata"
    if root_meta.exists():
        return _metadata_says_finished(root_meta)
    return False


def find_recordings(run_dir: Path, h5_glob: str, assay_subfolder: str = "Network") -> list[Path]:
    """Recordings that the driver would actually process.

    Mirrors ``helper_functions.find_files_with_subfolder``: in directory mode the
    driver only accepts ``data.raw.h5`` files that have ``assay_subfolder`` as a
    path component, which is how ActivityScan recordings get excluded.
    """
    if not assay_subfolder:
        return sorted(run_dir.rglob(h5_glob))
    return sorted(p for p in run_dir.rglob(h5_glob) if assay_subfolder in p.parts)


def find_recording(run_dir: Path, h5_glob: str, assay_subfolder: str = "Network") -> Optional[Path]:
    """First qualifying recording, falling back to any recording.

    The fallback covers flattened layouts (``<run>/data.raw.h5`` with no assay
    subfolder), which the driver can still handle in single-file mode.
    """
    hits = find_recordings(run_dir, h5_glob, assay_subfolder)
    if hits:
        return hits[0]
    direct = run_dir / h5_glob
    if direct.exists():
        return direct
    any_hit = sorted(run_dir.rglob(h5_glob))
    return any_hit[0] if any_hit else None


# --------------------------------------------------------------------------- #
# Watcher
# --------------------------------------------------------------------------- #
class Watcher:
    """Polls the watch directory and dispatches completed runs to the pipeline."""

    def __init__(self, cfg: JobConfig, on_event: Optional[Callable[[str, dict], None]] = None):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.work_dir / "watcher_state.json")
        self.on_event = on_event or (lambda *_: None)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # run_key -> (fingerprint, observed_at) from the previous poll
        self._prints: dict[str, tuple[tuple[int, int, float], float]] = {}

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mea-watcher", daemon=True)
        self._thread.start()
        LOG.info("Watcher started on %s", self.cfg.watch_dir)
        self.on_event("started", {"watch_dir": self.cfg.watch_dir})

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        LOG.info("Watcher stopped")
        self.on_event("stopped", {})

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:  # noqa: BLE001 — a bad scan must not kill the daemon
                LOG.exception("scan cycle failed; continuing")
            self._stop.wait(self.cfg.poll_seconds)

    # -- scanning ----------------------------------------------------------- #
    def jobs_for(self, run_dir: Path) -> list[str]:
        """Which enabled analyses actually have data in this folder."""
        jobs = []
        for job in self.cfg.enabled_jobs():
            if find_recordings(run_dir, self.cfg.h5_glob, self.cfg.subfolder_for(job)):
                jobs.append(job)
        return jobs

    def scan_candidates(self) -> list[tuple[Path, list[str]]]:
        """Run folders with the analyses that apply to each.

        Returns the job list alongside the folder so callers do not have to call
        ``jobs_for`` again — each call walks the tree with ``rglob``, which is
        expensive on multi-GB folders and was previously done twice per poll.
        """
        root = Path(self.cfg.watch_dir)
        if not root.is_dir():
            return []
        out: list[tuple[Path, list[str]]] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            jobs = self.jobs_for(child)
            if jobs:
                out.append((child, jobs))
        return out

    def candidate_runs(self) -> list[Path]:
        return [d for d, _ in self.scan_candidates()]

    @staticmethod
    def state_key(run_dir: Path, job: str) -> str:
        """Each analysis is tracked independently for the same folder."""
        return f"{run_dir}::{job}"

    def scan_once(self) -> None:
        for run_dir, jobs in self.scan_candidates():
            resolved = run_dir.resolve()
            pending = [j for j in jobs
                       if not self.state.is_claimed(self.state_key(resolved, j))]
            if not pending:
                continue

            # Completion is a property of the folder, so it is checked once and
            # shared by both analyses.
            ready, detail = self._check_ready(run_dir, str(resolved), jobs)
            for job in pending:
                key = self.state_key(resolved, job)
                if ready:
                    self.dispatch(run_dir, job, detail=detail)
                else:
                    self.state.update(key, status="waiting", detail=detail,
                                      job=job, run=run_dir.name, last_seen=_now())

    def _check_ready(self, run_dir: Path, key: str,
                     jobs: Optional[list[str]] = None) -> tuple[bool, str]:
        """Stateless-across-polls completion check (never blocks).

        ``jobs`` is passed in by the scan loop so the tree is not walked again.
        """
        if jobs is None:
            jobs = self.jobs_for(run_dir)
        # Check markers across every assay folder we are going to analyze, so a
        # still-recording activity scan holds back its own job too.
        subfolders = [self.cfg.subfolder_for(j) for j in jobs]
        if self.cfg.require_finished_marker and not all(
                has_finished_marker(run_dir, self.cfg.h5_glob, sf) for sf in subfolders or [None]):
            self._prints.pop(key, None)
            return False, "waiting for MaxWell 'finished' marker"

        now = time.time()
        current = folder_fingerprint(run_dir)
        previous = self._prints.get(key)

        # First sighting: record the fingerprint and start the clock.
        if previous is None:
            self._prints[key] = (current, now)
            return False, "observing — first fingerprint taken"

        prev_print, prev_at = previous

        # Anything changed => still copying; restart the settle clock.
        if current != prev_print:
            self._prints[key] = (current, now)
            gb = current[1] / 1e9 if current[1] > 0 else 0
            return False, f"still copying ({current[0]} files, {gb:.1f} GB)"

        # Unchanged: deliberately keep the ORIGINAL observation time so the
        # settle window actually elapses across polls.
        elapsed = now - prev_at
        if elapsed < self.cfg.settle_seconds:
            remaining = int(self.cfg.settle_seconds - elapsed)
            return False, f"stable, settling ({remaining}s remaining)"

        gb = current[1] / 1e9
        return True, f"complete ({current[0]} files, {gb:.1f} GB, stable {int(elapsed)}s)"

    # -- dispatch ----------------------------------------------------------- #
    def build_command(self, run_dir: Path, job: str = JOB_NETWORK) -> list[str]:
        """Build the command for one analysis of a completed run folder."""
        if job == JOB_ACTIVITY:
            return self._build_activity_command(run_dir)
        return self._build_network_command(run_dir)

    def _build_network_command(self, run_dir: Path) -> list[str]:
        """Spike-sorting pipeline (``run_pipeline_driver.py``).

        Prefer **directory mode**: the driver then discovers every qualifying
        recording (and every recording x well inside each file) itself, and
        applies its own ``Network`` filter. Passing a single file would analyze
        just that one recording.

        Fall back to single-file mode for flattened layouts, where no recording
        sits under the assay subfolder and directory mode would find nothing.
        """
        qualifying = find_recordings(run_dir, self.cfg.h5_glob, self.cfg.assay_subfolder)
        if qualifying:
            target = str(run_dir)
        else:
            recording = find_recording(run_dir, self.cfg.h5_glob, self.cfg.assay_subfolder)
            target = str(recording) if recording else str(run_dir)
        return [self.cfg.python, str(self.cfg.driver), target,
                *build_driver_args(self.cfg.driver_options)]

    def _build_activity_command(self, run_dir: Path) -> list[str]:
        """Whole-array activity extraction (``activity_scan.py``).

        Deliberately independent of the spike-sorting pipeline: its own script,
        its own output directory, no GPU, and no shared state.
        """
        cmd = [self.cfg.python, str(ACTIVITY_SCRIPT), str(run_dir),
               "--assay-subfolder", self.cfg.activity_subfolder,
               "--active-hz", str(self.cfg.activity_active_hz)]
        out = self.cfg.activity_out
        if out:
            cmd += ["--output-dir", out]
        if not self.cfg.activity_figures:
            cmd.append("--no-figures")

        # If a Network recording exists alongside, overlay which electrodes it
        # kept — that is the selection-bias view, and it is free to compute.
        net = find_recordings(run_dir, self.cfg.h5_glob, self.cfg.assay_subfolder)
        if net:
            cmd += ["--selection-from", str(net[0])]
        return cmd

    def dispatch(self, run_dir: Path, job: str = JOB_NETWORK, detail: str = "") -> None:
        key = self.state_key(run_dir.resolve(), job)
        label = JOB_LABELS.get(job, job)
        cmd = self.build_command(run_dir, job)
        printable = " ".join(shlex.quote(c) for c in cmd)
        LOG.info("Dispatching %s [%s]:\n    %s", run_dir.name, label, printable)

        common = {"job": job, "job_label": label, "run": run_dir.name,
                  "command": printable, "detail": detail}

        if self.cfg.dry_run:
            self.state.update(key, status="detected", detected_at=_now(), **{
                **common, "detail": f"dry run — {detail}" if detail else "dry run"})
            self.on_event("detected", {"run": run_dir.name, "job": job, "command": printable})
            return

        log_path = self.log_dir / f"{run_dir.name}_{job}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.state.update(key, status="dispatched", dispatched_at=_now(),
                          log=str(log_path), **common)
        self.on_event("dispatched", {"run": run_dir.name, "job": job, "log": str(log_path)})

        threading.Thread(
            target=self._run_job, args=(run_dir, job, key, cmd, log_path),
            name=f"mea-{job}-{run_dir.name}", daemon=True,
        ).start()

    def _run_job(self, run_dir: Path, job: str, key: str,
                 cmd: list[str], log_path: Path) -> None:
        label = JOB_LABELS.get(job, job)
        started = time.time()
        self.state.update(key, status="running", started_at=_now())
        self.on_event("running", {"run": run_dir.name, "job": job})
        try:
            with open(log_path, "w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
            ok = proc.returncode == 0
            self.state.update(
                key,
                status="done" if ok else "failed",
                completed_at=_now(),
                returncode=proc.returncode,
                duration_s=round(time.time() - started, 1),
            )
            (LOG.info if ok else LOG.error)(
                "%s [%s] finished with code %s (log: %s)",
                run_dir.name, label, proc.returncode, log_path)
            self.on_event("done" if ok else "failed",
                          {"run": run_dir.name, "job": job, "returncode": proc.returncode})
        except Exception as exc:  # noqa: BLE001
            self.state.update(key, status="failed", completed_at=_now(), error=str(exc))
            LOG.exception("%s [%s]: dispatch raised", run_dir.name, label)
            self.on_event("failed", {"run": run_dir.name, "job": job, "error": str(exc)})

    # -- status for the UI --------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        runs = []
        for key, entry in self.state.all().items():
            folder = key.split("::")[0]
            job = entry.get("job", JOB_NETWORK)
            runs.append({
                "path": key,
                "folder": folder,
                "run": entry.get("run", Path(folder).name),
                "job": job,
                "job_label": entry.get("job_label", JOB_LABELS.get(job, job)),
                **entry,
            })
        runs.sort(key=lambda r: (r.get("run", ""), r.get("job", "")))
        counts: dict[str, int] = {}
        by_job: dict[str, dict[str, int]] = {}
        for r in runs:
            st = r.get("status", "unknown")
            counts[st] = counts.get(st, 0) + 1
            by_job.setdefault(r["job"], {})[st] = by_job.setdefault(r["job"], {}).get(st, 0) + 1
        return {
            "running": self.is_running,
            "watch_dir": self.cfg.watch_dir,
            "output_dir": self.cfg.output_dir,
            "activity_output_dir": self.cfg.activity_out,
            "enabled_jobs": self.cfg.enabled_jobs(),
            "counts": counts,
            "counts_by_job": by_job,
            "runs": runs,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-config", type=Path, default=None,
                   help="JSON job config written by the UI")
    p.add_argument("--watch-dir", type=Path, default=None, help="Input path to watch (read-only)")
    p.add_argument("--output-dir", type=Path, default=None, help="Pipeline output path")
    p.add_argument("--config", type=Path, default=None, help="mea_config.json passed to the driver")
    p.add_argument("--settle-seconds", type=int, default=None)
    p.add_argument("--poll-seconds", type=int, default=None)
    p.add_argument("--work-dir", type=Path, default=None, help="Where state + logs are written")
    p.add_argument("--no-finished-marker", action="store_true")
    p.add_argument("--once", action="store_true", help="Scan once and exit")
    p.add_argument("--dry-run", action="store_true", help="Detect and log, do not launch")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = JobConfig.load(a.job_config) if a.job_config else JobConfig()
    if a.watch_dir:
        cfg.watch_dir = str(a.watch_dir)
    if a.output_dir:
        cfg.driver_options["output_dir"] = str(a.output_dir)
    if a.config:
        cfg.driver_options["config"] = str(a.config)
    if a.settle_seconds is not None:
        cfg.settle_seconds = a.settle_seconds
    if a.poll_seconds is not None:
        cfg.poll_seconds = a.poll_seconds
    if a.work_dir:
        cfg.work_dir = str(a.work_dir)
    if a.no_finished_marker:
        cfg.require_finished_marker = False
    if a.dry_run:
        cfg.dry_run = True

    errors = cfg.validate()
    if errors:
        raise SystemExit("Invalid configuration:\n  - " + "\n  - ".join(errors))

    watcher = Watcher(cfg)
    if a.once:
        # Two fingerprints separated by the FULL settle window, otherwise the
        # second scan can never conclude the folder is quiescent.
        watcher.scan_once()
        wait = cfg.settle_seconds + 1
        LOG.info("Waiting %ss for the settle window…", wait)
        time.sleep(wait)
        watcher.scan_once()
        LOG.info("-" * 60)
        for run in watcher.snapshot()["runs"]:
            LOG.info("%-12s %-11s %s", run.get("run"), run.get("status"), run.get("detail", ""))
        return

    watcher.start()
    try:
        while watcher.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
