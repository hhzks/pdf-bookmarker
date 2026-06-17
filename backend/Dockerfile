FROM python:3.12-slim-bookworm

WORKDIR /srv

COPY pyproject.toml ./
COPY pdf_bookmarker ./pdf_bookmarker
RUN pip install --no-cache-dir ".[gemini,web]"

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

COPY backend/app ./app

# Render injects PORT; default for local runs.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
