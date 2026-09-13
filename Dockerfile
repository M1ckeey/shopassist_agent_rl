FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_backend ./llm_backend

WORKDIR /app/llm_backend

ENV DB_TYPE=sqlite
ENV SQLITE_PATH=/app/data/assistgen.db
ENV UPLOAD_DIR=/app/data/uploads
ENV GRAPHRAG_PROJECT_DIR=/app/llm_backend/app/graphrag

RUN mkdir -p /app/data/uploads

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-9000}"]
