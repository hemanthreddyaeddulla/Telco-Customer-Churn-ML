# Telco Customer Churn Prediction

[![CI](https://github.com/hemanthreddyaeddulla/Telco-Customer-Churn-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/hemanthreddyaeddulla/Telco-Customer-Churn-ML/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
[![demo](https://img.shields.io/badge/demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/hemanthreddy901/telco-churn)

Predicting which telecom customers are likely to cancel their service, and turning each
prediction into a concrete retention decision. This is a complete project: data
exploration, a clean training pipeline, a calibrated and explainable model, a test suite
with a CI quality gate, basic monitoring, and a deployed web app.

**Live demo:** https://huggingface.co/spaces/hemanthreddy901/telco-churn

## Problem

A telecom company loses revenue when customers leave. Keeping an existing customer is much
cheaper than acquiring a new one, so the business wants to (1) predict who is likely to
churn and (2) decide who is worth contacting with a retention offer. This project answers
both questions, not just the first.

## Dataset

The public IBM Telco Customer Churn dataset: 7,043 customers and 21 columns covering
demographics, the services each customer subscribes to, and account and billing
information, with a binary `Churn` label. About 26.5% of customers churned. The data is
not committed to the repository; download it with `python scripts/fetch_data.py`, which
saves it to `data/raw/`.

## Approach

1. Exploratory analysis (`notebooks/telco_churn_end_to_end.ipynb`): class balance, a data
   quality check that traced the blank `TotalCharges` values to tenure-0 customers,
   correlations with churn, and a VIF check showing the "no internet service" dummy
   columns are perfectly collinear (a reason to prefer tree models).
2. Cleaning: drop the customer ID, convert `TotalCharges` to numeric (the tenure-0 blanks
   are filled with 0), and map the target to 0/1.
3. Feature transforms inside one scikit-learn `Pipeline`: median imputation for the
   numeric columns and one-hot encoding for the categoricals, fit once and reused.
4. Model: RandomForest, LightGBM and XGBoost were compared; XGBoost was chosen and tuned
   with Optuna. Its probabilities are calibrated with isotonic regression.
5. Decision threshold: rather than a fixed 0.5, the threshold is chosen to maximise the
   expected retention value given cost assumptions (customer value, offer cost, save rate).
6. Explainability: every prediction returns the top SHAP drivers and a suggested action.
7. Serving: a FastAPI service with a Gradio web interface, packaged with Docker.

## Key design decisions

- One fitted pipeline for both training and serving. The saved model is a single
  `Pipeline` that accepts the raw customer columns, so the serving code does not
  re-implement any feature logic. This avoids a subtle problem where encoding one row at a
  time with `get_dummies(drop_first=True)` collapses a category to its dropped reference
  (for example, a fiber-optic customer being scored as DSL). On the holdout set that naive
  approach would change the prediction for roughly 1 in 4 customers.
- Calibrated probabilities. `scale_pos_weight` helps the tree handle the class imbalance
  but inflates its probabilities; isotonic calibration maps them back, so a predicted 34%
  means an actual 34%. With calibrated probabilities the cost-based threshold lands at the
  economic break-even point.
- A pandera schema validates incoming data (allowed category values, numeric ranges) and
  is also reused as a test fixture.
- A CI quality gate. The GitHub Actions workflow runs ruff, mypy and the test suite, then
  trains a quick model and fails the build if ROC-AUC or recall fall below a committed
  floor, so a change that quietly degrades the model breaks the build.
- Monitoring. Predictions are logged to a file, and `scripts/drift_report.py` computes a
  Population Stability Index (PSI) drift report against the training distribution.
- A model registry. `scripts/register_model.py` and `scripts/promote_model.py` implement a
  champion/challenger promotion gate on top of the MLflow Model Registry.

## Results

- RandomForest, LightGBM and XGBoost all reach about 0.83 ROC-AUC on the holdout set.
- The Optuna-tuned, calibrated XGBoost reaches about 0.85 ROC-AUC, catching about 85% of
  churners at the chosen threshold.
- The goal here is not a higher AUC (calibration does not change ranking). The value is in
  serving correctness, calibrated probabilities, a cost-based decision, and explanations.

## Repository structure

```
src/
  data/load_data.py        data loading
  features/pipeline.py     clean_data + the preprocessing-and-model pipeline
  models/threshold.py      cost-based decision threshold
  models/tune.py           Optuna hyperparameter tuning
  models/registry.py       MLflow Model Registry config
  serving/inference.py     load the model, predict, SHAP explanations
  serving/monitoring.py    prediction logging + PSI drift
  app/main.py              FastAPI + Gradio app
scripts/
  fetch_data.py            download the dataset
  run_pipeline.py          train end to end and log to MLflow
  check_metrics.py         CI model-quality gate
  drift_report.py          PSI drift report
  register_model.py        register a challenger model
  promote_model.py         champion/challenger promotion gate
  deploy_hf.py             deploy the app to Hugging Face Spaces
notebooks/
  telco_churn_end_to_end.ipynb   full walkthrough, EDA to deployment
tests/                     unit, data-contract, behavioural, serving and monitoring tests
```

## Running it locally

```bash
conda create -n ml_churn python=3.11 -y
conda activate ml_churn
pip install -r requirements.txt

python scripts/fetch_data.py                 # downloads the dataset to data/raw/
python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
python -m uvicorn src.app.main:app --port 8000
# open http://localhost:8000 for the app and http://localhost:8000/docs for the API
```

Quality checks:

```bash
pytest
ruff check .
mypy src
python scripts/check_metrics.py
```

## Deployment

The app is containerised (`dockerfile`); the image downloads the data and trains the model
at build time, then serves it. A GitHub Actions workflow builds and publishes the image to
Docker Hub. The public demo is hosted for free on Hugging Face Spaces (created with
`scripts/deploy_hf.py`).

## Tech stack

Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandera, MLflow, FastAPI, Gradio,
pytest, ruff, mypy, Docker, GitHub Actions.

## Limitations and next steps

- The retention economics (customer value, offer cost, save rate) are assumptions that
  control how aggressively customers are flagged; in practice they should be set with the
  business.
- The reported numbers come from a single train/test split; cross-validation would add
  confidence intervals.
- The free demo runs on a small CPU instance and sleeps when idle, then wakes on the next
  visit.
