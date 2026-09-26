FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY floodmark_pipeline ./floodmark_pipeline
RUN pip install .

CMD ["arq", "floodmark_pipeline.worker.WorkerSettings"]
