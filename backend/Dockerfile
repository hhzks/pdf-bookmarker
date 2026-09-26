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

# The model for PDFs not produced by TeX (Word, InDesign, report generators).
# The model above was trained on 87% LaTeX and scores 0.4221 title F1 on
# held-out web PDFs, below the font heuristics; this one scores 0.6531 there.
# Same release/digest discipline: a retrain of either is a new asset plus a
# bump of its own two ARGs.
ARG LABELER_NONTEX_VERSION=labeler-nontex-v1
ARG LABELER_NONTEX_SHA256=4be3044292ec9d8839b1246063776ee15fc2b15f0877f9e4c178ca7c7e70b9dc
RUN curl -fsSL -o /srv/models/labeler-nontex.joblib \
        "https://github.com/hhzks/pdf-bookmarker/releases/download/${LABELER_NONTEX_VERSION}/labeler-nontex.joblib" \
    && echo "${LABELER_NONTEX_SHA256}  /srv/models/labeler-nontex.joblib" | sha256sum -c -

# Defaults for a plain `docker run`; render.yaml declares them too, so the
# deployment's configuration is readable without opening the Dockerfile.
ENV PDF_BOOKMARKER_LABELER=/srv/models/labeler.joblib
ENV PDF_BOOKMARKER_LABELER_NONTEX=/srv/models/labeler-nontex.joblib

COPY backend/app ./app

# Render injects PORT; default for local runs.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
