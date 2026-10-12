# REST API image (CPU).
#   docker build -t vietner .
#   docker run -p 8000:8000 vietner          # http://localhost:8000/docs
# PhoBERT is downloaded at build time (PRELOAD=true) so containers start offline.

FROM python:3.11-slim

ARG PRELOAD=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY vietner ./vietner
COPY training ./training
COPY models ./models
RUN pip install ".[api]"
RUN if [ "$PRELOAD" = "true" ]; then python -c "from vietner.encoder import WordEncoder; WordEncoder.from_pretrained()"; fi

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD \
    python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"
CMD ["vietner", "serve-api", "--port", "8000"]
