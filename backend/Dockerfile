FROM python:3.12-slim-bookworm

WORKDIR /srv

COPY pyproject.toml ./
COPY pdf_bookmarker ./pdf_bookmarker
# scikit-learn is pinned to the version that fitted the shipped model. The
# bundle is pickled estimators, and unpickling those across scikit-learn
# versions is not guaranteed to work — and the failure would land at boot, on
# a deploy that changed nothing but the build date. The [labeler] extra keeps
# its wider >=1.4 range for library users, who bring their own model.
RUN pip install --no-cache-dir ".[gemini,labeler,web]" "scikit-learn==1.8.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# The heading model ships as a release asset rather than a tracked file: it is
# ~5 MB of fitted estimators and a retrain replaces it wholesale, which is not
# something to accumulate in git history. Pinned by tag and verified by digest,
# so an image either contains the exact model that was measured or fails to
# build. Without this the server falls back to font heuristics and serves
# 0.6208 title F1 in place of 0.8009, while looking perfectly healthy.
ARG LABELER_VERSION=labeler-v1
ARG LABELER_SHA256=bb9039a3516f95b392f0b273dfaa8ca660b893e0fe108a7f7002e0fe6341ec21
RUN mkdir -p /srv/models \
    && curl -fsSL -o /srv/models/labeler.joblib \
        "https://github.com/hhzks/pdf-bookmarker/releases/download/${LABELER_VERSION}/labeler.joblib" \
    && echo "${LABELER_SHA256}  /srv/models/labeler.joblib" | sha256sum -c -

# Default for a plain `docker run`; render.yaml declares it too, so the
# deployment's configuration is readable without opening the Dockerfile.
ENV PDF_BOOKMARKER_LABELER=/srv/models/labeler.joblib

COPY backend/app ./app

# Render injects PORT; default for local runs.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
