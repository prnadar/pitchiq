FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models if not already present
RUN python -c "from pathlib import Path; \
    saved = Path('backend/models/saved'); \
    saved.mkdir(parents=True, exist_ok=True); \
    xgb = saved / 'xgb_model.pkl'; \
    exec('from backend.models.train import train_and_evaluate; train_and_evaluate()') if not xgb.exists() else None"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
