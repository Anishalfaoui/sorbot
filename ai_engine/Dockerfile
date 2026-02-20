# ── Sorbot AI Engine (Spot Trading) ──────────
FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data/models dirs
RUN mkdir -p data models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
