# Deploying the MEA orchestration pipeline on a new server

Step-by-step, with a verification after each stage. Every step assumes you are
logged into the target server as the user that will own the analysis output.

---

## 0. Prerequisites

```bash
# Docker and the Compose v2 plugin
docker --version
docker compose version

# You must be able to run docker without sudo
docker ps

# NVIDIA Container Toolkit (needed by Kilosort4)
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**Expected:** the last command prints your GPU table from *inside* a container.

If `docker ps` fails with a permission error, ask an admin to add you to the
`docker` group. If the `--gpus all` run fails, the NVIDIA Container Toolkit is
missing — the UI will still work, but analysis will fail later.

---

## 1. Clone the repository

```bash
cd ~                                    # or wherever you keep code
git clone https://github.com/hiteshkumar18/MEA-Analysis.git
cd MEA-Analysis
git log --oneline -1                    # confirm you have the latest commit
git remote -v                           # confirm origin is your fork
```

---

## 2. Decide your paths and record your IDs

```bash
id -u          # -> HOST_UID
id -g          # -> HOST_GID
pwd            # -> HOST_REPO_DIR
```

You need three directories:

| Purpose | Requirement |
|---|---|
| **Input** | Contains recording run folders (e.g. `000041`). Mounted read-only. |
| **Output** | Where results are written. Must be writable by you. |
| **Work** | Watcher state, job config, run logs. Must be writable by you. |

---

## 3. Create the directories BEFORE starting Docker

This is the single most common failure. Docker creates missing bind-mount
directories **as root**, and without sudo you cannot fix the ownership
afterwards.

```bash
mkdir -p ~/AnalyzedData          # output
mkdir -p .mea-watcher            # work dir (inside the repo)

ls -ld ~/AnalyzedData .mea-watcher     # both must show YOUR username, not root
```

Also confirm your input folder exists and is readable:

```bash
ls -l /path/to/your/recordings   # should list run folders like 000041
```

---

## 4. Configure `.env`

```bash
cp .env.example .env
nano .env
```

Set at minimum:

```
HOST_INPUT_DIR=/path/to/your/recordings        # PARENT of the run folders
HOST_OUTPUT_DIR=/home/<you>/AnalyzedData
HOST_WORK_DIR=/home/<you>/MEA-Analysis/.mea-watcher
HOST_REPO_DIR=/home/<you>/MEA-Analysis
UI_PORT=8000
HOST_UID=<id -u>
HOST_GID=<id -g>
```

Use **absolute paths** — `~` is not expanded inside `.env`.

Verify nothing is left unset:

```bash
grep -v '^#' .env | grep -v '^$'
```

If port 8000 is taken on this server, pick another:

```bash
ss -tlnp | grep :8000        # if this prints anything, change UI_PORT
```

---

## 5. Build the base image

This is the long step (CUDA, Kilosort, SpikeInterface). Expect 10–30 minutes.

```bash
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
docker images | grep mea-spikesorter
```

To build from a different fork or branch:

```bash
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile \
  --build-arg MEA_REPO_URL=https://github.com/<you>/<repo>.git \
  --build-arg MEA_REPO_BRANCH=<branch> .
```

---

## 6. Build and start the orchestration container

```bash
docker compose up -d --build
docker compose ps
```

**Expected:** `STATUS` shows `Up … (healthy)` and `PORTS` shows
`0.0.0.0:8000->8000/tcp`.

```bash
docker compose logs --tail 20
```

**Expected:** the startup banner with your repo, work dir, input, and output
paths, then "Open this in your browser".

If the container is not running, read the logs — the entrypoint reports
permission problems explicitly and tells you how to fix them.

---

## 7. Verify the container internals

```bash
# GPU visible inside the container
docker compose exec mea-orchestration nvidia-smi

# The bind-mounted repo is present and is YOUR code
docker compose exec mea-orchestration ls -l "$(grep '^HOST_REPO_DIR=' .env | cut -d= -f2)/run_pipeline_driver.py"

# Input data is visible and read-only
docker compose exec mea-orchestration ls "$(grep '^HOST_INPUT_DIR=' .env | cut -d= -f2)"

# Output is writable by the container user
docker compose exec mea-orchestration touch "$(grep '^HOST_OUTPUT_DIR=' .env | cut -d= -f2)/.probe" \
  && docker compose exec mea-orchestration rm "$(grep '^HOST_OUTPUT_DIR=' .env | cut -d= -f2)/.probe" \
  && echo "output writable OK"

# API responds
curl -s -o /dev/null -w "UI: HTTP %{http_code}\n" http://localhost:8000/
curl -s http://localhost:8000/api/status | head -c 200; echo
```

---

## 8. Reach the UI from your laptop

The UI has **no authentication**, so do not expose it on the network. Tunnel it
over SSH instead. Run this **on your local machine**:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<server>
```

Leave it running, then open <http://localhost:8000>.

Use `127.0.0.1` rather than `localhost` in the `-L` argument — on some servers
`localhost` resolves to IPv6 first and the forward is refused. If your local
port 8000 is busy, map a different one: `-L 9000:127.0.0.1:8000`.

---

## 9. Dry run — verify detection without launching anything

Confirm the watcher sees your data and builds the right command.

**From the command line:**

```bash
docker compose exec mea-orchestration \
  python3 orchestration/watcher.py \
    --watch-dir "$(grep '^HOST_INPUT_DIR=' .env | cut -d= -f2)" \
    --output-dir "$(grep '^HOST_OUTPUT_DIR=' .env | cut -d= -f2)" \
    --settle-seconds 20 --poll-seconds 10 \
    --work-dir /tmp/dryrun-state \
    --once --dry-run -v
```

**Expected:** the exact command it would run, then a summary line such as

```
000041   detected   dry run — complete (198 files, 6.5 GB, stable 21s)
```

Use a throwaway `--work-dir` (as above). A dry run marks the run as claimed, so
reusing the real work dir would cause the actual run to be skipped.

**Or from the UI:** set the input and output paths, enable **Dry run**, click
**Preview command**, then **Start watching** and confirm the run reaches
"Detected" without launching.

---

## 10. First real run

1. In the UI, turn **Dry run** off.
2. Review the pipeline options (sorter, curation, plotting).
3. Set the **settle window** for production — 600s is a sane default for
   multi-GB copies over a network.
4. Click **Start watching**.

Any run folder already present and finished will be picked up immediately, so
expect analysis to begin about one settle window later.

Watch progress in the Runs table, or:

```bash
docker compose logs -f
ls -R "$(grep '^HOST_OUTPUT_DIR=' .env | cut -d= -f2)" | head
nvidia-smi                      # confirm the GPU is actually busy
```

---

## Day-to-day operation

```bash
docker compose ps               # status
docker compose logs -f          # live logs
docker compose restart          # restart the service
docker compose down             # stop
git pull && docker compose restart   # update code (no rebuild needed)
```

The repo is bind-mounted and the container sets `MEA_SKIP_UPDATE=1`, so the code
that runs is whatever is in your checkout. Update with `git pull`; only changes
to the Dockerfile or entrypoint require `docker compose up -d --build`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `mkdir: cannot create directory '…': Permission denied` | Bind-mount dir is root-owned. `docker compose down`, recreate it as yourself (step 3), start again. |
| Container restarts repeatedly | Read `docker compose logs`; the entrypoint names the problem. |
| `address already in use` on start | Another process holds `UI_PORT`. Change it in `.env`. |
| SSH tunnel: `connection refused` | Nothing listening on the server yet — check `docker compose ps`, and make sure the tunnel port matches `UI_PORT`. |
| Browser: "Failed to fetch" | The HTML was opened as a file. Open `http://localhost:<port>` instead. |
| UI: "Input path does not exist" | The path is not mounted into the container, or you pointed at a single run instead of its parent. |
| Runs never leave "waiting" | Still copying, or the MaxWell `finished=` marker is absent. Toggle that requirement off in the UI if your copy process strips it. |
| Runs reprocessed after restart | `HOST_WORK_DIR` was not persisted. |
| Results owned by root | `HOST_UID`/`HOST_GID` do not match `id -u`/`id -g`. |
| Kilosort CUDA errors | GPU not passed through — recheck step 0. |
