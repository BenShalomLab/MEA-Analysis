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

    # Detection tuning
    h5_glob: str = "data.raw.h5"
    # Only recordings under this path component are processed, matching the
    # driver's own filter. "Network" excludes ActivityScan. Blank = no filter.
    assay_subfolder: str = "Network"
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
        if not Path(self.driver).exists():
            errs.append(f"run_pipeline_driver.py not found at: {self.driver}")
        out = self.output_dir
        if out:
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errs.append(f"Output path is not writable: {out} ({exc})")
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

    def get(self, key: str) -> dict:
        return self._data.get(key, {})

    def status(self, key: str) -> Optional[str]:
        return self.get(key).get("status")

    def is_claimed(self, key: str) -> bool:
        return self.status(key) in self.CLAIMED

    def update(self, key: str, **fields) -> None:
        with self._lock:
            entry = self._data.setdefault(key, {"run": Path(key).name})
            entry.update(fields)
            self._flush()

    def all(self) -> dict[str, dict]:
        return dict(self._data)

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
    def candidate_runs(self) -> list[Path]:
        root = Path(self.cfg.watch_dir)
        if not root.is_dir():
            return []
        out = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and find_recording(child, self.cfg.h5_glob, self.cfg.assay_subfolder):
                out.append(child)
        return out

    def scan_once(self) -> None:
        for run_dir in self.candidate_runs():
            key = str(run_dir.resolve())
            if self.state.is_claimed(key):
                continue
            ready, detail = self._check_ready(run_dir, key)
            if ready:
                self.dispatch(run_dir, detail=detail)
            else:
                self.state.update(key, status="waiting", detail=detail, last_seen=_now())

    def _check_ready(self, run_dir: Path, key: str) -> tuple[bool, str]:
        """Stateless-across-polls completion check (never blocks)."""
        if self.cfg.require_finished_marker and not has_finished_marker(
                run_dir, self.cfg.h5_glob, self.cfg.assay_subfolder):
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
    def build_command(self, run_dir: Path) -> list[str]:
        """Build the driver invocation for a completed run folder.

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
        return [self.cfg.python, str(self.cfg.driver), target, *build_driver_args(self.cfg.driver_options)]

    def dispatch(self, run_dir: Path, detail: str = "") -> None:
        key = str(run_dir.resolve())
        cmd = self.build_command(run_dir)
        printable = " ".join(shlex.quote(c) for c in cmd)
        LOG.info("Dispatching %s:\n    %s", run_dir.name, printable)

        if self.cfg.dry_run:
            self.state.update(key, status="detected", detected_at=_now(),
                              command=printable, detail=f"dry run — {detail}" if detail else "dry run")
            self.on_event("detected", {"run": run_dir.name, "command": printable})
            return

        log_path = self.log_dir / f"{run_dir.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.state.update(key, status="dispatched", dispatched_at=_now(),
                          command=printable, log=str(log_path), detail=detail)
        self.on_event("dispatched", {"run": run_dir.name, "log": str(log_path)})

        threading.Thread(
            target=self._run_pipeline, args=(run_dir, key, cmd, log_path),
            name=f"mea-run-{run_dir.name}", daemon=True,
        ).start()

    def _run_pipeline(self, run_dir: Path, key: str, cmd: list[str], log_path: Path) -> None:
        started = time.time()
        self.state.update(key, status="running", started_at=_now())
        self.on_event("running", {"run": run_dir.name})
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
                "%s finished with code %s (log: %s)", run_dir.name, proc.returncode, log_path)
            self.on_event("done" if ok else "failed",
                          {"run": run_dir.name, "returncode": proc.returncode})
        except Exception as exc:  # noqa: BLE001
            self.state.update(key, status="failed", completed_at=_now(), error=str(exc))
            LOG.exception("%s: dispatch raised", run_dir.name)
            self.on_event("failed", {"run": run_dir.name, "error": str(exc)})

    # -- status for the UI --------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        runs = []
        for key, entry in self.state.all().items():
            runs.append({"path": key, "run": entry.get("run", Path(key).name), **entry})
        runs.sort(key=lambda r: r.get("run", ""))
        counts: dict[str, int] = {}
        for r in runs:
            counts[r.get("status", "unknown")] = counts.get(r.get("status", "unknown"), 0) + 1
        return {
            "running": self.is_running,
            "watch_dir": self.cfg.watch_dir,
            "output_dir": self.cfg.output_dir,
            "counts": counts,
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
