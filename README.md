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
saves it to `data/raw/`. Source:
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d).

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

## How a prediction becomes a decision

This is the core of the project, so it is worth spelling out end to end.

### Calibrated probabilities (why isotonic regression on a classifier)

XGBoost already outputs a number between 0 and 1, and log loss is the training objective that
pushes it toward probability-like values. So why add calibration? Because this project sets
`scale_pos_weight` to handle the class imbalance (~26% churn), which deliberately makes the
model over-predict churn. The result is a good *ranking* but an inflated *probability*: a raw
0.55 can correspond to a real churn rate of only ~34%. Log loss does not fix this, because the
class reweighting shifts the base rate the model effectively learns.

Isotonic regression here is a post-processing step, not the classifier itself. It learns a
single monotonic mapping (`raw score -> true probability`) on held-out folds. Because the
mapping is monotonic it never reorders customers, so ROC-AUC is unchanged; it only corrects the
values so that "the model says 34%" really means "about 34 out of 100 such customers churn".
That matters because the next step multiplies the probability by dollar amounts.

### The decision threshold (cost-based, derived, not a fixed 0.5)

A default 0.5 cut-off quietly assumes a false alarm costs the same as a missed churner. They are
very different, so each outcome is given a dollar value (`src/models/threshold.py`):

| Outcome | Meaning | Value |
|---|---|---|
| True positive  | flag a real churner   | `save_rate*CLV - cost` = +$100 |
| False positive | flag a loyal customer | `-cost` = -$200 |
| False negative | miss a churner        | `-CLV` = -$1000 |
| True negative  | correctly ignore      | $0 |

The training script computes the total expected dollar value at every candidate threshold (a
grid from 0.05 to 0.95) on a validation split and picks the one that maximises money. With these
costs that reduces to a simple break-even:

```
act on a customer when   p * CLV * (1 + save_rate) > cost
                  =>      p > cost / (CLV * (1 + save_rate)) = 200 / (1000 * 1.30) ~= 0.15
```

which is why the chosen threshold is about 0.14. It is low because missing a churner (-$1000) is
roughly five times more costly than a wasted offer (-$200), so it pays to contact a customer
even at a modest churn probability. That is also why recall is high (0.91 on the test set).

The threshold is **derived, not assumed**: change the economics (`--clv`, `--retention_cost`,
`--save_rate`) and it moves. It is computed once at training, saved in `feature_contract.json`,
and loaded at serving, so every request uses that same value (it does not recompute per request).

### Explanations

Every prediction returns the top SHAP drivers, aggregated per original feature and labelled with
the customer's actual value (for example "Contract = Month-to-month"), plus a retention action
tied to the strongest churn-increasing driver.

## Engineering and MLOps decisions

- One fitted pipeline for both training and serving. The saved model is a single
  `Pipeline` that accepts the raw customer columns, so the serving code does not
  re-implement any feature logic. This avoids a subtle problem where encoding one row at a
  time with `get_dummies(drop_first=True)` collapses a category to its dropped reference
  (for example, a fiber-optic customer being scored as DSL). On the holdout set that naive
  approach would change the prediction for roughly 1 in 4 customers.
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

All models are evaluated on the same untouched holdout set (1,409 customers, 20% of the data).

Model comparison (ROC-AUC on the holdout):

| Model                                | ROC-AUC |
|--------------------------------------|---------|
| RandomForest                         | 0.823   |
| LightGBM                             | 0.829   |
| XGBoost (default params)             | 0.825   |
| XGBoost (Optuna-tuned + calibrated)  | 0.848   |

Final model, train vs test (at the cost-optimal threshold of 0.14):

| Set            | ROC-AUC | Recall (churn) | Precision (churn) | F1    |
|----------------|---------|----------------|-------------------|-------|
| Train (n=5,634)| 0.877   | 0.933          | 0.463             | 0.619 |
| Test  (n=1,409)| 0.848   | 0.906          | 0.451             | 0.603 |

The small train-to-test gap (0.877 vs 0.848 ROC-AUC) shows the model is well regularized and
not overfit. The cost-based threshold deliberately favours recall (0.906 on the test set):
missing a churner who would have stayed is far more expensive than sending one unnecessary
retention offer, and the expected-value calculation makes that trade-off explicit.

The goal here is not a higher AUC (calibration does not change the ranking, so it does not
move ROC-AUC). The value is in serving correctness, calibrated probabilities you can trust as
real percentages, a cost-based decision, and per-prediction explanations.

The numbers above use the Optuna-tuned model (`run_pipeline.py --tune`, what the notebook
runs). The Docker build and the CI gate use the faster untuned run (`--no_tune`, fixed
default params), which scores about 0.834 ROC-AUC; the CI gate only checks the score stays
above a committed floor.

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

### Calling the API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
  "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "tenure": 1,
  "MonthlyCharges": 85.0, "TotalCharges": 85.0
}'
```

Response:

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.82,
  "threshold": 0.14,
  "top_drivers": [
    {"feature": "Contract = Month-to-month", "impact": 0.86, "direction": "increases churn"},
    {"feature": "tenure", "impact": 0.84, "direction": "increases churn"},
    {"feature": "InternetService = Fiber optic", "impact": 0.33, "direction": "increases churn"}
  ],
  "recommended_action": "Offer a discounted 1- or 2-year contract to lock in the customer."
}
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
at build time, then serves it. A GitHub Actions workflow (`deploy.yml`) builds and publishes
the image to Docker Hub. The public Hugging Face Spaces demo then **pulls that pre-built image**
(`scripts/deploy_hf.py --image <user>/telco-churn:latest`), so the exact same artifact is served
everywhere (build once, deploy anywhere) instead of retraining on the demo host.

## Tech stack

Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandera, MLflow, FastAPI, Gradio,
pytest, ruff, mypy, Docker, GitHub Actions.

## Limitations and next steps

- The retention economics (customer value, offer cost, save rate) are assumptions that
  control how aggressively customers are flagged; in practice they should be set with the
  business.
- The reported numbers come from a single train/test split; cross-validation would add
  confidence intervals.
- The cost-based threshold is recomputed from the validation set on every training run, and
  gradient boosting is not bitwise-deterministic across machines, so the exact value can shift
  slightly by environment (about 0.14 locally, 0.21 in the published image) even though the model
  quality (ROC-AUC ~0.85) is unchanged. The shipped Docker image is one fixed artifact, so every
  deployment of it behaves identically.
- The free demo runs on a small CPU instance and sleeps when idle, then wakes on the next
  visit.

## License

Released under the MIT License. See [LICENSE](LICENSE).
