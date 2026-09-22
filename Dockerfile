# VectoVecto - document restore web app (CPU image).
#
# The image ships the product code and the OCR models that come with the
# rapidocr wheel; it never contains training data, checkpoints or evaluation
# corpora (see .dockerignore).
#
#   docker build -t vectovecto .
#   docker run -p 7860:7860 vectovecto
#
# Optional basic auth:
#   docker run -p 7860:7860 -e VECTOVECTO_USER=me -e VECTOVECTO_PASSWORD=secret vectovecto

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VECTOVECTO_HOST=0.0.0.0 \
    VECTOVECTO_PORT=7860 \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py cli.py ./
COPY calibration.py degradation_document.py doc_data.py doc_metrics.py ./
COPY document_export.py document_layout.py document_ocr.py \
     document_orientation.py document_pipeline.py document_restore.py \
     document_router.py document_verifier.py ./
COPY smart_upscaler.py sr_engine.py srvggnet.py tv_refinement.py \
     vector_raster_hybrid.py ./

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; \
urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ['VECTOVECTO_PORT'], timeout=4)"

CMD ["python", "app.py"]
