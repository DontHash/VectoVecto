# VeriScript — document restore web app (CPU image).
#
# Multi-stage: the SolidJS frontend is built with Node, then served by the
# FastAPI backend. The image ships the product code and the OCR models that
# come with the rapidocr wheel; it never contains training data, checkpoints
# or evaluation corpora (see .dockerignore).
#
#   docker build -t veriscript .
#   docker run -p 8000:8000 veriscript
#
# Optional basic auth for /api/*:
#   docker run -p 8000:8000 -e VERISCRIPT_WEB_USER=me \
#     -e VERISCRIPT_WEB_PASSWORD=secret veriscript

# -- frontend: build the studio once ------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /web
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

# -- backend -------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VERISCRIPT_WEB_HOST=0.0.0.0 \
    VERISCRIPT_WEB_PORT=8000 \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Document-only image: no torch (the web document path never imports it);
# the photo stack and the opt-in readers stay local/full-install features.
COPY requirements-doc.txt webapp/requirements-web.txt ./
RUN pip install -r requirements-doc.txt -r requirements-web.txt

COPY cli.py ./
COPY degradation_document.py doc_data.py doc_metrics.py ./
COPY document_export.py document_layout.py document_ocr.py \
     document_orientation.py document_pipeline.py document_restore.py \
     document_router.py document_verifier.py ./
COPY smart_upscaler.py sr_engine.py srvggnet.py tv_refinement.py \
     vector_raster_hybrid.py ./
COPY lexicon.py logging_setup.py branding.py ./
COPY calibration.py ./

# Bundled OFL font for the Devanagari PDF text layer and overlay annotations.
COPY fonts/ fonts/
# Isotonic calibration JSON behind `cal_conf` (kept out of .dockerignore).
COPY calibration/ calibration/

# Web app: FastAPI backend + the frontend built in the first stage.
COPY webapp/server.py webapp/server.py
COPY webapp/vvweb/ webapp/vvweb/
COPY --from=frontend /web/dist webapp/frontend/dist

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p webapp/runs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; \
urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['VERISCRIPT_WEB_PORT'], timeout=4)"

CMD ["python", "webapp/server.py"]
