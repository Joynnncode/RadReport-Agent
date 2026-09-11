# syntax=docker/dockerfile:1
#
# CPU-only image, for local reproducibility and the Azure Container Apps
# deployment (docs/azure-deploy.md). No GPU, no CUDA.
#
# Two decisions worth explaining:
#
#   1. torch comes from PyTorch's CPU wheel index. The default PyPI wheel for
#      Linux bundles CUDA and pulls roughly 2 GB of libraries this project never
#      touches -- every model here runs on CPU.
#
#   2. Model weights are baked in at build time. torchxrayvision downloads
#      DenseNet and PSPNet (~100 MB) on first use; without this the first request
#      in a fresh container silently blocks on a network fetch, and in an air-
#      gapped environment it fails outright. Better to pay it once at build.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    XRV_CACHE=/opt/xrv

WORKDIR /app

# libgomp1 is required by torch; the rest of build-essential is not needed for
# wheels, so we stay on slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install -r requirements.txt

# Pre-download model weights so the first request does not block on a fetch.
# Same reasoning for the sentence encoder: it is only used by the non-default
# retriever, but "only sometimes hangs on a network fetch" is worse than always.
ENV SENTENCE_TRANSFORMERS_HOME=/opt/sbert
RUN python -c "\
import torchxrayvision as xrv; \
xrv.models.DenseNet(weights='densenet121-res224-all'); \
xrv.baseline_models.chestx_det.PSPNet(); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('weights cached')"

# Weights are baked in above, so nothing should ever reach the Hub at runtime.
# Without this the first embedding call in an air-gapped container still spends
# ~100 seconds on HEAD requests for OPTIONAL config files, each retried five
# times, before falling back to the cache and working. The pre-download step
# prevented the failure and not the hang, which is the more insidious half:
# nothing errors, the request is just inexplicably slow, once.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY radreport/ ./radreport/
COPY evals/ ./evals/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY app.py pytest.ini README.md DECISIONS.md requirements-deploy.txt ./

# The two data files git already ships ARE copied in: the report corpus (1.3 MB)
# and the demo cache (10 MB). The image used to find them only because compose
# mounts ./data over the top; on a host with nothing to mount, retrieval was empty.
#
# The X-ray images are NOT. They are gitignored, 300 MB, and baking a dataset
# into a public image is how licence terms get violated by accident. They are
# mounted at data/images instead: ./data via compose locally, an Azure Files
# share in the cloud (docs/azure-deploy.md). And not the demo cache's thumbnails
# written back out either: measured, those move classifier probabilities by up
# to 0.23 against the full-resolution originals. See DECISIONS.md, 2026-09-11.
COPY data/reports.csv data/demo_cache.json ./data/
RUN mkdir -p data/images artifacts traces .cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
