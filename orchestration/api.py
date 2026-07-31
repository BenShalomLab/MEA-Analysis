#!/usr/bin/env python3
"""
FastAPI backend for the MEA pipeline control UI.

Serves the single-page frontend and exposes the watcher as a REST API:

    GET  /api/schema        every run_pipeline_driver.py option (drives the form)
    GET  /api/config        current saved job config
    POST /api/config        save job config (validates paths)
    POST /api/browse        list subdirectories of a server path (path picker)
    POST /api/preview       show the exact command that will be run
    GET  /api/status        watcher state + per-run status (polled by the UI)
    POST /api/watcher/start start watching
    POST /api/watcher/stop  stop watching
    POST /api/runs/reset    forget a run so it can be re-processed
    GET  /api/runs/log      tail a run's pipeline log

Run:
    pip install fastapi uvicorn
    python orchestration/api.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver_schema import schema_for_ui, validate_options, default_options  # noqa: E402
from watcher import (  # noqa: E402
    JobConfig, Watcher, DEFAULT_WORK_DIR, find_recording, find_recordings, has_finished_marker,
)

LOG = logging.getLogger("mea.api")
HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "static" / "index.html"


class RingLogHandler(logging.Handler):
    """Keeps recent log records in memory so the UI can stream watcher activity.

    Each record gets a monotonic sequence number; the client polls with the last
    sequence it saw and receives only what is new.
    """

    def __init__(self, maxlen: int = 2000):
        super().__init__()
        self.records: deque = deque(maxlen=maxlen)
        self.seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.seq += 1
            self.records.append({
                "seq": self.seq,
                "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:  # noqa: BLE001 - logging must never raise
            pass

    def since(self, seq: int, limit: int = 500) -> list[dict]:
        return [r for r in self.records if r["seq"] > seq][-limit:]


LOG_RING = RingLogHandler()

app = FastAPI(title="MEA Pipeline Control", version="1.0")

# --------------------------------------------------------------------------- #
# Process-wide state
# --------------------------------------------------------------------------- #
WORK_DIR = DEFAULT_WORK_DIR
CONFIG_PATH = WORK_DIR / "job.json"
_watcher: Optional[Watcher] = None
_events: list[dict] = []


def _record_event(kind: str, payload: dict) -> None:
    _events.append({"kind": kind, **payload})
    del _events[:-200]  # keep the last 200


def load_job_config() -> JobConfig:
    if CONFIG_PATH.exists():
        try:
            return JobConfig.load(CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not load %s (%s); using defaults", CONFIG_PATH, exc)
    return JobConfig(work_dir=str(WORK_DIR))


def get_watcher() -> Watcher:
    global _watcher
    if _watcher is None:
        _watcher = Watcher(load_job_config(), on_event=_record_event)
    return _watcher


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ConfigPayload(BaseModel):
    watch_dir: str = ""
    driver_options: dict[str, Any] = {}
    h5_glob: str = "data.raw.h5"
    assay_subfolder: str = "Network"
    settle_seconds: int = 600
    poll_seconds: int = 30
    require_finished_marker: bool = True
    dry_run: bool = False


class BrowsePayload(BaseModel):
    path: str = ""


class RunKeyPayload(BaseModel):
    path: str


# --------------------------------------------------------------------------- #
# Routes — schema & config
# --------------------------------------------------------------------------- #
@app.get("/api/schema")
def api_schema():
    return {"groups": schema_for_ui(), "defaults": default_options()}


def _runtime_env() -> dict:
    """Detect containerization so the UI can warn about container-vs-host paths."""
    in_container = (
        Path("/.dockerenv").exists()
        or os.environ.get("MEA_REPO", "").startswith("/MEA_Analysis")
    )
    return {
        "in_container": in_container,
        "suggested_input": os.environ.get("MEA_INPUT_DIR"),
        "suggested_output": os.environ.get("MEA_OUTPUT_DIR"),
    }


@app.get("/api/env")
def api_env():
    return _runtime_env()


@app.get("/api/config")
def api_get_config():
    cfg = get_watcher().cfg
    env = _runtime_env()
    # First run inside a container: prefill the mount points so nobody types
    # host paths that don't exist in here.
    if not cfg.watch_dir and env["suggested_input"]:
        cfg.watch_dir = env["suggested_input"]
    if not cfg.driver_options.get("output_dir") and env["suggested_output"]:
        cfg.driver_options["output_dir"] = env["suggested_output"]
    return {
        "env": env,
        "watch_dir": cfg.watch_dir,
        "driver_options": cfg.driver_options,
        "h5_glob": cfg.h5_glob,
        "assay_subfolder": cfg.assay_subfolder,
        "settle_seconds": cfg.settle_seconds,
        "poll_seconds": cfg.poll_seconds,
        "require_finished_marker": cfg.require_finished_marker,
        "dry_run": cfg.dry_run,
        "work_dir": cfg.work_dir,
    }


@app.post("/api/config")
def api_set_config(payload: ConfigPayload):
    watcher = get_watcher()
    if watcher.is_running:
        raise HTTPException(409, "Stop the watcher before changing configuration.")

    opt_errors = validate_options(payload.driver_options)
    if opt_errors:
        raise HTTPException(400, {"errors": opt_errors})

    opts = default_options()
    opts.update(payload.driver_options)

    cfg = JobConfig(
        watch_dir=payload.watch_dir,
        driver_options=opts,
        h5_glob=payload.h5_glob,
        assay_subfolder=payload.assay_subfolder,
        settle_seconds=payload.settle_seconds,
        poll_seconds=payload.poll_seconds,
        require_finished_marker=payload.require_finished_marker,
        work_dir=str(WORK_DIR),
        dry_run=payload.dry_run,
    )
    errors = cfg.validate()
    if errors:
        raise HTTPException(400, {"errors": errors})

    cfg.save(CONFIG_PATH)
    global _watcher
    _watcher = Watcher(cfg, on_event=_record_event)

    # "detected" is only ever produced by a dry run, and it counts as claimed —
    # so a run marked that way would never be analyzed for real. Clear those
    # entries when leaving dry-run mode, otherwise turning the switch off
    # appears to do nothing.
    cleared = 0
    if not cfg.dry_run:
        for key, entry in list(_watcher.state.all().items()):
            if entry.get("status") == "detected":
                _watcher.state.reset(key)
                cleared += 1
        if cleared:
            LOG.info("Dry run disabled — cleared %d detected run(s) for real analysis", cleared)

    return {"ok": True, "saved_to": str(CONFIG_PATH), "cleared_detected": cleared}


# --------------------------------------------------------------------------- #
# Routes — helpers
# --------------------------------------------------------------------------- #
@app.post("/api/browse")
def api_browse(payload: BrowsePayload):
    """List subdirectories so the UI can offer a server-side path picker."""
    raw = payload.path.strip() or str(Path.home())
    p = Path(raw).expanduser()
    if not p.exists():
        raise HTTPException(404, f"Path does not exist: {p}")
    if not p.is_dir():
        p = p.parent
    try:
        entries = sorted(
            (c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")),
            key=lambda c: c.name.lower(),
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {p}")

    cfg = get_watcher().cfg
    items = []
    for c in entries:
        recs = find_recordings(c, cfg.h5_glob, cfg.assay_subfolder)
        rec = recs[0] if recs else find_recording(c, cfg.h5_glob, cfg.assay_subfolder)
        items.append({
            "name": c.name,
            "path": str(c),
            "is_run": rec is not None,
            "recordings": len(recs) or (1 if rec else 0),
            "finished": (has_finished_marker(c, cfg.h5_glob, cfg.assay_subfolder)
                         if rec is not None else None),
        })
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": items}


@app.post("/api/preview")
def api_preview(payload: ConfigPayload):
    """Show the exact command the watcher will execute for a detected run."""
    opts = default_options()
    opts.update(payload.driver_options)
    cfg = JobConfig(watch_dir=payload.watch_dir, driver_options=opts,
                    h5_glob=payload.h5_glob, assay_subfolder=payload.assay_subfolder,
                    work_dir=str(WORK_DIR))
    probe = Watcher(cfg, on_event=lambda *_: None)
    runs = probe.candidate_runs()
    sample = runs[0] if runs else Path(payload.watch_dir or "/path/to") / "000000"
    import shlex
    return {
        "detected_runs": [r.name for r in runs],
        "command": " ".join(shlex.quote(c) for c in probe.build_command(sample)),
        "example_run": sample.name,
    }


# --------------------------------------------------------------------------- #
# Routes — watcher control & status
# --------------------------------------------------------------------------- #
@app.post("/api/watcher/start")
def api_start():
    watcher = get_watcher()
    errors = watcher.cfg.validate()
    if errors:
        raise HTTPException(400, {"errors": errors})
    watcher.start()
    return {"ok": True, "running": watcher.is_running}


@app.post("/api/watcher/stop")
def api_stop():
    watcher = get_watcher()
    watcher.stop()
    return {"ok": True, "running": watcher.is_running}


@app.get("/api/status")
def api_status():
    watcher = get_watcher()
    snap = watcher.snapshot()
    snap["events"] = _events[-25:]
    snap["candidates"] = [c.name for c in watcher.candidate_runs()]
    return snap


@app.post("/api/runs/reset")
def api_reset(payload: RunKeyPayload):
    watcher = get_watcher()
    watcher.state.reset(payload.path)
    watcher._prints.pop(payload.path, None)
    return {"ok": True}


@app.get("/api/runs/log")
def api_log(path: str, tail: int = 400):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Log not found")
    lines = p.read_text(errors="ignore").splitlines()
    return {"path": str(p), "lines": lines[-tail:], "size": p.stat().st_size}


@app.get("/api/logs")
def api_logs(since: int = 0, limit: int = 500):
    """Live watcher activity — polled by the UI with the last seq it received."""
    return {"lines": LOG_RING.since(since, limit), "last_seq": LOG_RING.seq}


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    if not FRONTEND.exists():
        return JSONResponse({"error": f"Frontend not found at {FRONTEND}"}, status_code=500)
    return FileResponse(FRONTEND)


def main() -> None:
    global WORK_DIR, CONFIG_PATH
    p = argparse.ArgumentParser(description="MEA pipeline control UI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                   help="Where job config, watcher state, and logs are written")
    a = p.parse_args()

    WORK_DIR = a.work_dir
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = WORK_DIR / "job.json"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    # Capture watcher + API activity for the UI's live log panel.
    LOG_RING.setLevel(logging.INFO)
    for name in ("mea.watcher", "mea.api"):
        logging.getLogger(name).addHandler(LOG_RING)

    if not FRONTEND.exists():
        raise SystemExit(f"Frontend missing: {FRONTEND}\n"
                         "Expected orchestration/static/index.html next to api.py.")

    # Fail fast with a clear message if the port is taken.
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((a.host, a.port))
    except OSError as exc:
        raise SystemExit(f"Cannot bind {a.host}:{a.port} — {exc}\n"
                         f"Another process may be using it. Try --port {a.port + 1}.")
    finally:
        probe.close()

    url_host = "localhost" if a.host in ("0.0.0.0", "127.0.0.1") else a.host
    LOG.info("Work dir: %s", WORK_DIR)
    LOG.info("=" * 58)
    LOG.info("  Open this in your browser:  http://%s:%s", url_host, a.port)
    LOG.info("  (do not open index.html directly — it must be served)")
    LOG.info("=" * 58)

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
