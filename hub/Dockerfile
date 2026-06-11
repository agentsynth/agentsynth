# Build from the repo root so the packs ship with the image:
#   docker build -f hub/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv
COPY hub/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hub/app ./app
COPY packs ./packs
ENV PACKS_DIR=/srv/packs
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
