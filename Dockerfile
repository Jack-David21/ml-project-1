FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts ./scripts
COPY data/processed/meta.json data/processed/meta.json
COPY models/lgbm_final.joblib models/lgbm_final.joblib
EXPOSE 8000
CMD ["uvicorn", "scripts.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]