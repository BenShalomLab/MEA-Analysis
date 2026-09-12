#!/bin/bash
set -euo pipefail

REPO_DIR="${MEA_REPO_DIR:-/MEA_Analysis}"
BRANCH="${MEA_REPO_BRANCH:-main}"
# Set MEA_SKIP_UPDATE=1 to run exactly the code already present (e.g. when the
# repo is bind-mounted from the host). Without this the reset below would
# discard uncommitted local changes.
SKIP_UPDATE="${MEA_SKIP_UPDATE:-0}"

cd "$REPO_DIR"

if [[ "$SKIP_UPDATE" != "1" ]]; then
    echo "=== Updating $(basename "$REPO_DIR") (branch: $BRANCH) before run ==="
    echo "    origin: $(git remote get-url origin 2>/dev/null || echo unknown)"
    git fetch origin "$BRANCH" --depth=1 || true
    git reset --hard origin/"$BRANCH" || true
else
    echo "=== Skipping repo update (MEA_SKIP_UPDATE=1) ==="
fi

echo "=== RUNNING PRE-PIPELINE HOOKS ==="
#excute docker patches before running the pipeline
if [ -f "$REPO_DIR/dockers/spikesorter/docker_patches.sh" ]; then
    echo "Executing docker patches..."
    chmod +x "$REPO_DIR/dockers/spikesorter/docker_patches.sh"
    REPO_DIR="$REPO_DIR" bash "$REPO_DIR/dockers/spikesorter/docker_patches.sh"
fi

# Locate the driver: it sits at the repo root in the current layout, with the
# older IPNAnalysis/ location kept as a fallback.
if [ -f "$REPO_DIR/run_pipeline_driver.py" ]; then
    DRIVER="$REPO_DIR/run_pipeline_driver.py"
elif [ -f "$REPO_DIR/IPNAnalysis/run_pipeline_driver.py" ]; then
    DRIVER="$REPO_DIR/IPNAnalysis/run_pipeline_driver.py"
else
    echo "ERROR: run_pipeline_driver.py not found under $REPO_DIR" >&2
    exit 1
fi

echo "=== Running pipeline ($DRIVER) ==="
exec python3 "$DRIVER" "$@"
