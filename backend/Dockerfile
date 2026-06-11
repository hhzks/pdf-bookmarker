FROM python:3.12-slim

WORKDIR /srv

COPY pyproject.toml ./
COPY pdf_bookmarker ./pdf_bookmarker
RUN pip install --no-cache-dir ".[gemini,web]"

COPY backend/app ./app

# Render injects PORT; default for local runs.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
