FROM python:3.11-slim

# libgomp1 is the OpenMP runtime that XGBoost and LightGBM link against.
# python:*-slim omits it, and without it importing them fails at runtime with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Retrains only if the committed models are missing or were written by a
# different library version (see scripts/ensure_models.py).
RUN python scripts/ensure_models.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
