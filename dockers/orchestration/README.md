# Running the orchestration layer in Docker

The orchestration image layers on top of the existing `mea-spikesorter` image,
so it inherits CUDA, Kilosort4, SpikeInterface, and the Maxwell HDF5
compression plugin. The watcher launches `run_pipeline_driver.py` as a local
subprocess **inside the same container** — no Docker socket, no sibling
containers.

## Deploying to a new server from scratch

```bash
# 1. Clone your fork
git clone https://github.com/hiteshkumar18/MEA-Analysis.git
cd MEA-Analysis

# 2. Configure this machine (paths, UID/GID, port)
cp .env.example .env
nano .env                 # NOT committed — .env is gitignored

# 3. Create the directories yourself, so Docker doesn't create them as root
mkdir -p .mea-watcher
mkdir -p "$(grep '^HOST_OUTPUT_DIR=' .env | cut -d= -f2)"

# 4. Build the base image, then bring everything up
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
docker compose up -d --build
```

Open **http://localhost:${UI_PORT}** (tunnel with
`ssh -N -L 8000:127.0.0.1:8000 user@server` if the server is remote).

### Which repo the images use

Both images clone from `MEA_REPO_URL` / `MEA_REPO_BRANCH`, which default to
this fork (`hiteshkumar18/MEA-Analysis`, `main`) and can be overridden in
`.env`. To build against a different fork or branch without editing files:

```bash
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile \
  --build-arg MEA_REPO_URL=https://github.com/<you>/<repo>.git \
  --build-arg MEA_REPO_BRANCH=<branch> .
```

At **runtime** the orchestration container sets `MEA_SKIP_UPDATE=1`, because the
repo is bind-mounted from your checkout. Without it the inherited entrypoint
would run `git reset --hard origin/main` and discard uncommitted local changes.
So the code that actually runs is whatever is in your clone — update it with
`git pull`, not by rebuilding.

## Identity mounts — paths are the same inside and out

Every directory is mounted at **the same path** inside the container as it has
on the host:

| `.env` variable | Mounted at | Mode |
|---|---|---|
| `HOST_INPUT_DIR` | same path | read-only |
| `HOST_OUTPUT_DIR` | same path | read-write |
| `HOST_WORK_DIR` | same path | read-write |
| `HOST_REPO_DIR` | same path | read-write |

So a path you type in the UI, read in a log, or copy out of a command works
identically on the server — no translation. Only mounted paths are visible
inside the container; anything else genuinely does not exist there.

**The input path must be the folder that *contains* run folders**, not a single
run. If your recordings are at `/home/you/MEA/000041`, set
`HOST_INPUT_DIR=/home/you/MEA`. The watcher scans that folder so it can pick up
each new run as it lands; folders without a recording file are ignored (so the
repo checkout sitting alongside the data is harmless).

Use **Browse** and **Preview command** in the UI to confirm before starting.

## Why the mounts are set up this way

* **Input is mounted `:ro`.** The requirement that we never write to the data
  server is enforced by Docker itself, not just by convention.
* **`/work` is a persistent volume.** It holds the job config, watcher state,
  and per-run logs. If you don't persist it, the watcher forgets which runs it
  already processed and will re-run them after a restart.
* **The repo is bind-mounted** at `/MEA_Analysis` so code changes apply without
  a rebuild. Comment that volume out to run purely from the image.

## Modes

```bash
# Control UI (default)
docker compose up -d

# Headless watcher — no browser, uses the job config saved by the UI
docker compose run --rm mea-orchestration watch

# Debug shell inside the container
docker compose run --rm mea-orchestration shell
```

## GPU, memory, and permissions

* GPU access uses the NVIDIA Container Toolkit. Verify with:
  `docker compose run --rm mea-orchestration shell -c "nvidia-smi"`
* `shm_size: 8gb` — PyTorch dataloaders fail with Docker's 64 MB default.
* `user: ${HOST_UID}:${HOST_GID}` — without this, results are written to your
  output folder owned by root. Set these to `id -u` and `id -g`.

## Permission denied on /work (first-run gotcha)

Docker creates missing bind-mount directories **as root**. The container runs
as `HOST_UID:HOST_GID`, so it then can't write its state or logs, and the
container exits.

```bash
docker compose down
id -u ; id -g                                  # put these in .env
sudo chown -R $(id -u):$(id -g) ./.mea-watcher
docker compose up -d
```

Check the output directory for the same problem — it fails much later, in the
middle of a run:

```bash
ls -ld /your/output/path
```

Creating both directories yourself *before* the first `docker compose up`
avoids this entirely.

## Watching bind-mounted / network volumes

Completion detection polls file sizes and mtimes rather than using inotify.
That's deliberate: inotify events are unreliable across bind mounts, NFS, and
CIFS, which is exactly where recording data lives. Polling works everywhere at
the cost of a small, configurable delay.

If the input is a slow network share, raise the settle window in the UI so a
mid-transfer stall isn't mistaken for completion.

## Troubleshooting

| Symptom | Cause |
|---|---|
| "Failed to fetch" in the browser | UI opened as a file, or container not running. Open `http://localhost:8000`. |
| Port unreachable from the host | Server must bind `0.0.0.0` inside the container — the entrypoint does this; don't override `MEA_UI_HOST` with `127.0.0.1`. |
| "Input path does not exist" | You entered a host path. Use `/data/incoming`. |
| Results owned by root | Set `HOST_UID` / `HOST_GID` in `.env`. |
| `mkdir: cannot create directory '/work/logs': Permission denied` | The host work dir is root-owned (Docker creates missing bind-mount dirs as root). Fix below. |
| Runs reprocessed after restart | `/work` wasn't persisted. |
| Kilosort fails with CUDA errors | GPU not passed through; check the NVIDIA Container Toolkit. |
| h5 read errors | `HDF5_PLUGIN_PATH` must point at the Maxwell plugin (set in the base image). |
