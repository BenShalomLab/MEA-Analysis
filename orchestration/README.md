# orchestration/

A control UI + watcher that automates step 1 of the MEA pipeline:

> data lands on the analysis server → wait until the dump is **completely
> written** → run `run_pipeline_driver.py` with the options set in the UI

## Design constraints

* **Read-only on the data server.** The watcher only reads run folders (file
  sizes, mtimes, and the MaxWell `mxassay.metadata` marker). It never writes
  anything into the watched directory.
* **The analysis repo is never modified.** `run_pipeline_driver.py` is invoked
  as a plain subprocess, exactly as a human would run it.
* **The UI is the only control surface.** Input path, output path, and every
  driver flag are set in the browser.
* **Nothing is written into the repo.** Job config, watcher state, and per-run
  logs go to a work directory (default `~/.mea-watcher`, override `--work-dir`).

## Files

| File | Purpose |
|---|---|
| `driver_schema.py` | Single source of truth for all 39 `run_pipeline_driver.py` options. Drives both the UI form and command construction, so they cannot drift. |
| `watcher.py` | Completion detection + dispatch. Usable standalone from the CLI. |
| `activity_scan.py` | Whole-array ActivityScan extraction — see [ACTIVITY_SCAN.md](ACTIVITY_SCAN.md). |
| `api.py` | FastAPI backend serving the UI and exposing the watcher as REST. |
| `static/index.html` | Single-file React frontend (CDN, no build step). |

## The two analyses

A completed folder can trigger either or both, chosen in the UI. They are kept
deliberately separate: different scripts, different outputs, independent status,
and one can fail without affecting the other.

| | Network | Activity scan |
|---|---|---|
| Script | `run_pipeline_driver.py` | `orchestration/activity_scan.py` |
| Reads | `<chip>/Network/<run>/data.raw.h5` | `<chip>/ActivityScan/<run>/data.raw.h5` |
| Does | Kilosort4 sorting, curation, bursts | Whole-array activity maps + QC |
| Needs | GPU; hours per chip | CPU only; seconds per chip |
| Output | `--output-dir` | `<output>/ActivityScan` (or its own path) |

Each folder therefore appears as **one row per enabled analysis** in the Runs
table, labelled Network or Activity scan, with its own status, log, and reset.

When both are enabled, the activity job automatically passes
`--selection-from` pointing at the Network recording, so its maps show which
electrodes were actually kept.

## Quick start

### Docker (recommended)

```bash
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
docker build -t mea-orchestration -f dockers/orchestration/Dockerfile .
cp .env.example .env && $EDITOR .env
docker compose up -d
# open http://localhost:8000
```

Inside the container, enter the **container** paths `/data/incoming` and
`/data/output` — not the host paths. See
[dockers/orchestration/README.md](../dockers/orchestration/README.md).

### Bare metal

```bash
pip install fastapi uvicorn
python orchestration/api.py --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

The page must be **served** by this API — opening `static/index.html` directly
in a browser gives "Failed to fetch", because the frontend has no server to
talk to.

In the UI: set the **input path** (where run folders are copied) and the
**output path** (`--output-dir`), adjust any pipeline options, then
**Start watching**. Use **Preview command** to see the exact command that will
run, and **Dry run** to detect without launching.

## How "completely written" is decided

A run folder is dispatched only when all of these hold:

1. It contains a recording file (`data.raw.h5` by default).
2. MaxWell wrote a `finished=` marker into `mxassay.metadata`
   (toggleable in the UI).
3. **Quiescence** — nothing anywhere under the folder changed in file count,
   total size, or mtime across two consecutive polls separated by the settle
   window (default 600s).

Detection never blocks: the previous poll's fingerprint is remembered, so many
runs can be supervised at once. Verified behavior on a file still being
appended to:

```
t+ 2s  waiting     still copying (2 files, 0.0 GB)
t+10s  waiting     still copying (2 files, 0.0 GB)
t+12s  waiting     stable, settling (2s remaining)
t+16s  dispatched  complete (2 files, stable 6s)
t+20s  done
```

Each run is recorded in `watcher_state.json` and dispatched **exactly once**;
the watcher is safe to restart. Use **Reset** in the UI to re-process a run.

### Tuning the settle window

The default 600s suits multi-GB copies over a network. Shorten it for fast
local copies; lengthen it if a copy can stall mid-transfer for longer than the
window (a stall longer than the settle window would otherwise look like
completion). The MaxWell `finished=` marker guards against this too — keep it
enabled when available.

## CLI (without the UI)

```bash
# one-shot dry run
python orchestration/watcher.py --watch-dir /mnt/incoming \
    --output-dir /data/AnalyzedData --config mea_config.json \
    --settle-seconds 600 --once --dry-run

# daemon, driven by a job config saved from the UI
python orchestration/watcher.py --job-config ~/.mea-watcher/job.json
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | All driver options (drives the form) |
| `GET` / `POST /api/config` | Read / save job config |
| `POST /api/browse` | Server-side folder picker; marks which folders are runs |
| `POST /api/preview` | Exact command that will be executed |
| `GET /api/status` | Watcher state + per-run status (UI polls every 3s) |
| `POST /api/watcher/start` / `stop` | Control the watcher |
| `POST /api/runs/reset` | Forget a run so it can be re-processed |
| `GET /api/runs/log` | Tail a run's pipeline log |

## Dependencies

`watcher.py` and `driver_schema.py` are standard library only.
The UI needs `fastapi` and `uvicorn`.

## Next step

Report/PDF generation consumes what the pipeline writes under `--output-dir`
(`well*/network_results.json`, metrics, raster figures). Not built yet.
