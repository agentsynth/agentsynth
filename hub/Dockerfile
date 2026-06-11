# Build from the repo root so the packs ship with the image:
#   docker build -f hub/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv

# curl is here so the platform's HTTP healthcheck can probe /healthz from inside the container
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY hub/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hub/app ./app
COPY packs ./packs
ENV PACKS_DIR=/srv/packs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
