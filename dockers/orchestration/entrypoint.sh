#!/usr/bin/env bash
# Entrypoint for the MEA orchestration container.
#
# Modes:
#   ui     (default)  serve the control UI  -> http://<host>:8000
#   watch             headless watcher, driven by the saved job config
#   shell             drop into bash for debugging
#
# Any other arguments are executed verbatim.

set -euo pipefail

REPO="${MEA_REPO:-/MEA_Analysis}"
WORK="${MEA_WATCHER_HOME:-/work}"
PORT="${MEA_UI_PORT:-8000}"
# Must bind 0.0.0.0 inside a container, otherwise the port is unreachable
# from the host even when published.
HOST="${MEA_UI_HOST:-0.0.0.0}"

# --- Preflight: writable work dir -------------------------------------------
# /work is a bind mount. If the host directory was created by Docker it is
# owned by root, and this container (running as HOST_UID:HOST_GID) cannot write
# to it. Fail fast with instructions instead of restart-looping.
if ! mkdir -p "${WORK}/logs" 2>/dev/null || [[ ! -w "${WORK}" ]]; then
  cat >&2 <<EOF
==============================================================
ERROR: cannot write to the work directory ${WORK}

It is bind-mounted from HOST_WORK_DIR on the host (default
./.mea-watcher) and is not writable by this container's user
($(id -u):$(id -g)).

Docker creates missing bind-mount directories as root, which is
the usual cause. On the host, run:

    docker compose down
    sudo chown -R \$(id -u):\$(id -g) ./.mea-watcher
    docker compose up -d

Also confirm HOST_UID / HOST_GID in .env match 'id -u' / 'id -g'.
==============================================================
EOF
  exit 1
fi

# Output dir must be writable too, or the pipeline fails much later.
if [[ -n "${MEA_OUTPUT_DIR:-}" && -d "${MEA_OUTPUT_DIR}" && ! -w "${MEA_OUTPUT_DIR}" ]]; then
  echo "WARNING: output dir ${MEA_OUTPUT_DIR} is not writable by $(id -u):$(id -g)." >&2
  echo "         Fix ownership on the host or results will fail to save." >&2
fi

cd "${REPO}"

if [[ ! -f "${REPO}/run_pipeline_driver.py" ]]; then
  echo "ERROR: run_pipeline_driver.py not found in ${REPO}" >&2
  echo "       Bind-mount the repo there, or rebuild the base image." >&2
  exit 1
fi

if [[ ! -f "${REPO}/orchestration/api.py" ]]; then
  echo "ERROR: orchestration/ missing in ${REPO}" >&2
  exit 1
fi

banner() {
  echo "=============================================================="
  echo "  MEA orchestration container"
  echo "  repo:      ${REPO}"
  echo "  work dir:  ${WORK}   (job config, watcher state, run logs)"
  echo "  input:     ${MEA_INPUT_DIR:-<not set>}   (mount read-only)"
  echo "  output:    ${MEA_OUTPUT_DIR:-<not set>}"
  echo "--------------------------------------------------------------"
  echo "  NOTE: paths typed in the UI are paths INSIDE this container."
  echo "=============================================================="
}

case "${1:-ui}" in
  ui)
    banner
    echo "  UI on http://localhost:${PORT} (published port on the host)"
    exec python3 orchestration/api.py --host "${HOST}" --port "${PORT}" --work-dir "${WORK}"
    ;;

  watch)
    banner
    shift || true
    if [[ -f "${WORK}/job.json" ]]; then
      echo "  headless watcher using ${WORK}/job.json"
      exec python3 orchestration/watcher.py --job-config "${WORK}/job.json" "$@"
    fi
    if [[ -z "${MEA_INPUT_DIR:-}" ]]; then
      echo "ERROR: no ${WORK}/job.json and MEA_INPUT_DIR is unset." >&2
      echo "       Configure once via the UI, or set MEA_INPUT_DIR/MEA_OUTPUT_DIR." >&2
      exit 1
    fi
    echo "  headless watcher from environment"
    exec python3 orchestration/watcher.py \
      --watch-dir "${MEA_INPUT_DIR}" \
      ${MEA_OUTPUT_DIR:+--output-dir "${MEA_OUTPUT_DIR}"} \
      ${MEA_CONFIG_FILE:+--config "${MEA_CONFIG_FILE}"} \
      ${MEA_SETTLE_SECONDS:+--settle-seconds "${MEA_SETTLE_SECONDS}"} \
      ${MEA_POLL_SECONDS:+--poll-seconds "${MEA_POLL_SECONDS}"} \
      --work-dir "${WORK}" "$@"
    ;;

  shell)
    exec /bin/bash
    ;;

  *)
    exec "$@"
    ;;
esac
