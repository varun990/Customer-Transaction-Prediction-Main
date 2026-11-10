# PRCP-1003 — Customer Transaction Prediction

Machine learning project that predicts which bank customers will make a specific transaction in the future,
irrespective of the amount transacted.

- **Domain:** Banking
- **Type:** Binary classification (`target`: 0 = no transaction, 1 = transaction)
- **Data:** 200,000 customers × 200 anonymized numeric features (`var_0` … `var_199`) + `ID_code`

## Architecture

```
Client → HTTP POST /predict → FastAPI → saved ML pipeline (.pkl) → transaction probability + prediction
```

## Project structure

```
├── data/
│   ├── raw/customer_transaction.csv        # original dataset (~300 MB)
│   └── processed/processed_data.csv        # scaled features + target + train/test split marker
├── models/
│   └── customer_transaction_model.pkl      # pipeline (preprocessing + XGBoost), features, threshold
├── reports/
│   ├── figures/                            # every chart produced by the notebook
│   └── Customer_Transaction_Report.pdf     # final report (17 sections)
├── deployment/
│   ├── app.py                              # FastAPI prediction service
│   ├── example_request.json                # sample request body
│   └── requirements.txt                    # minimal API dependencies
├── tests/
│   └── test_prediction.py                  # pytest: model loads → input accepted → prediction → probability
├── Customer Transaction Prediction.ipynb   # the complete analysis (Steps 1–23, one notebook)
├── requirements.txt                        # full Python environment
└── README.md
```

## Notebook steps

1. Import libraries → 2. Load dataset → 3. Understand dataset → 4. Missing values → 5. Duplicates → 6. Identify target
→ 7. EDA (target distribution, numerical / categorical analysis, outliers, correlation)
→ 8. Preprocessing (duplicates, missing values, X / y, split, numerical + categorical pipelines, ColumnTransformer)
→ 9. Logistic Regression → 10. Random Forest → 11. XGBoost → 12. Evaluate and compare (which metric?)
→ 13. Confusion matrix → 14. Cross-validation → 15. Hyperparameter tuning → 16. Evaluate final model
→ 17. Feature importance → 18. Save model → 19. Load saved model → 20. New prediction → 21. Deployment
→ 22. Testing → 23. Final report

## Results

| Model | Test ROC-AUC | 5-fold CV ROC-AUC | Recall | F1 |
|-------|-------------:|------------------:|-------:|---:|
| **XGBoost (tuned)** — recommended | **0.8945** | **0.8941** | 0.760 | 0.511 |
| XGBoost | 0.8871 | 0.8877 | 0.757 | 0.491 |
| Logistic Regression (baseline) | 0.8599 | 0.8590 | 0.775 | 0.418 |
| Random Forest | 0.8279 | 0.8324 | 0.185 | 0.286 |

(Precision / recall / F1 at threshold 0.5.) The final tuned XGBoost uses an F1-optimal decision threshold of **0.683**,
chosen on out-of-fold training predictions. On the test set this gives accuracy 0.909, precision 0.546, recall 0.573 and
F1 0.559 for the transaction class.

## Setup

```bash
pip install -r requirements.txt
```

On macOS, XGBoost also needs the OpenMP runtime (`brew install libomp`).

Put the dataset at `data/raw/customer_transaction.csv`, then open `Customer Transaction Prediction.ipynb` in Jupyter
and run all cells (about 25 minutes, mostly cross-validation and tuning).

## Run the API

```bash
uvicorn deployment.app:app --reload
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check and model info |
| `/predict` | POST | Score one customer: `{"features": {"var_0": 11.2, ..., "var_199": -3.1}}` |
| `/predict/batch` | POST | Score many: `{"customers": [{"features": {...}}, ...]}` |

Interactive docs: http://127.0.0.1:8000/docs

## Run the tests

```bash
python -m pytest tests -v
```
