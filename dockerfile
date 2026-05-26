# Telco churn serving image: FastAPI + Gradio over the trained sklearn pipeline.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the project.
COPY . .

# mlflow uses GitPython to read a commit SHA; this slim image has no git, so quiet the warning.
ENV GIT_PYTHON_REFRESH=quiet

# Fetch the full dataset (not stored in the repo) and bake a trained model into the image
# at /app/model (+ the threshold contract at /app/). Add --tune to run Optuna (slower build).
RUN python scripts/fetch_data.py \
    && python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --no_tune \
    && MODEL_SRC="$(ls -dt mlruns/*/*/artifacts/model | head -1)" \
    && cp -r "$MODEL_SRC" /app/model \
    && cp artifacts/feature_contract.json /app/feature_contract.json

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/model

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
