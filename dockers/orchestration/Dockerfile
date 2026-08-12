# MEA orchestration UI + watcher
#
# Layers on top of the existing spikesorter image so it inherits CUDA, Kilosort,
# SpikeInterface, and the Maxwell HDF5 compression plugin. The watcher launches
# run_pipeline_driver.py as a local subprocess inside this same container, so
# everything the pipeline needs is already present.
#
# Build the base first (from the repo root):
#     docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
#
# Then build this:
#     docker build -t mea-orchestration -f dockers/orchestration/Dockerfile .
#
# If you prefer a standalone image that does NOT depend on the base, override:
#     --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
# (that path installs the full requirements.txt and takes much longer)

ARG BASE_IMAGE=mea-spikesorter:latest
FROM ${BASE_IMAGE}

# Recorded for reference; the running container uses the bind-mounted checkout.
ARG MEA_REPO_URL=https://github.com/hiteshkumar18/MEA-Analysis.git
ARG MEA_REPO_BRANCH=main
LABEL org.opencontainers.image.source=${MEA_REPO_URL}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEA_WATCHER_HOME=/work

# Web layer only — everything else comes from the base image.
RUN pip3 install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27"

# Repo location inside the container. The base image clones the project to
# /MEA_Analysis; compose bind-mounts your working tree over it for development.
ENV MEA_REPO=/MEA_Analysis
WORKDIR ${MEA_REPO}

# Ship the orchestration layer so the image works without a bind mount too.
COPY orchestration/ ${MEA_REPO}/orchestration/

# Work dir holds job config, watcher state, and per-run logs.
# Mount a volume here so state survives container restarts.
RUN mkdir -p /work/logs

COPY dockers/orchestration/entrypoint.sh /usr/local/bin/mea-entrypoint.sh
RUN chmod +x /usr/local/bin/mea-entrypoint.sh

EXPOSE 8000

# Lightweight health probe for compose/orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4).status==200 else 1)" \
        || exit 1

ENTRYPOINT ["/usr/local/bin/mea-entrypoint.sh"]
CMD ["ui"]
