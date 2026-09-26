FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
ENV PYTHONPATH=/app/floodml_src \
    FLOODML_MODEL_DIR=/models
COPY backend/pyproject.toml ./
COPY backend/floodmark_pipeline ./floodmark_pipeline
COPY flood-ml/src ./floodml_src
COPY backend/onnx/floodmark_models_v4 /models
RUN pip install .

CMD ["arq", "floodmark_pipeline.worker.WorkerSettings"]
