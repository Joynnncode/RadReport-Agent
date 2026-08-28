# syntax=docker/dockerfile:1
#
# CPU-only image for local reproducibility. No GPU, no CUDA.
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

# Data is NOT copied in. It is gitignored, it is 500 MB of medical images, and
# baking a dataset into an image is how licence terms get violated by accident.
# docker-compose mounts ./data instead; fetch it on the host first.
RUN mkdir -p data/images artifacts traces .cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
