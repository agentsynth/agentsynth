# Build from the repo root so the library installs from source:
#   docker build -f playground/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv

# curl is here so the platform's HTTP healthcheck can probe the app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY agentsynth ./agentsynth
RUN pip install --no-cache-dir ".[app]"

COPY app.py .

# Offline mock by default; add a provider key (ANTHROPIC_API_KEY, ...) in the
# platform's env settings to switch the generator and judge to a real LLM.
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1
CMD ["python", "app.py"]
