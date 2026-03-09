# Universal ML Template 

A lightweight, multi-purpose ML template for **tabular datasets** (Kaggle-friendly):
- **Classification** and **Regression**
- Automatic preprocessing (numeric + categorical)
- Cross-validation over multiple models
- Automatic **best-model selection**
- Optional regression target transform (`log1p` with inverse transform for submission)
- Optional complex models via **builders** (e.g., stacking)

---

## Install & Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Quick setup (edit `config.py`)

### House Prices (Regression)

```python
DATA_CONFIG["target"] = "SalePrice"
DATA_CONFIG["id_col"] = "Id"

TRAIN_CONFIG["task"] = "regression"
TRAIN_CONFIG["metric"] = "rmse"
TRAIN_CONFIG["direction"] = "min"
TRAIN_CONFIG["target_transform"] = "log1p"
TRAIN_CONFIG["prediction_column"] = "SalePrice"

SELECTED_MODELS = ["cat_regressor", "stacking_hp", "xgb_regressor", "ridge"]
```

### Titanic (Classification)

```python
DATA_CONFIG["target"] = "Survived"
DATA_CONFIG["id_col"] = "PassengerId"

TRAIN_CONFIG["task"] = "classification"
TRAIN_CONFIG["metric"] = "accuracy"
TRAIN_CONFIG["direction"] = "max"
TRAIN_CONFIG["target_transform"] = None
TRAIN_CONFIG["prediction_column"] = "Survived"

SELECTED_MODELS = ["rf_classifier", "cat_classifier", "logreg", "xgb_classifier"]
```

---

## Outputs

After running `python main.py`, the template saves:

- `artifacts/models/model.joblib` — trained best model 
- `artifacts/reports/best_report.json` — CV report (folds, mean/std, leaderboard)
- `artifacts/submissions/submission.csv` — Kaggle-ready submission 

---

## Project structure

- `main.py` — entry point:
  - loads data
  - runs CV for `SELECTED_MODELS`
  - selects best model using `TRAIN_CONFIG["metric"]` and `TRAIN_CONFIG["selection"]`
  - trains the best model on full train
  - saves artifacts + optional submission

- `config.py` — all configuration:
  - `DATA_CONFIG`: paths, target column, id column, columns to drop
  - `TRAIN_CONFIG`: task, metric, CV settings, selection strategy, target transform
  - `MODEL_CANDIDATES`: model registry (standard + builder-based)
  - `SELECTED_MODELS`: which models to compare in this run

- `data.py` — data helpers:
  - `read_csv`
  - `drop_cols_safe`
  - `get_xy`

- `features.py` — dataset-specific feature engineering (must work identically for train and test)

- `pipeline.py` — sklearn pipeline builder:
  - numeric pipeline: imputer + optional scaler
  - categorical pipeline: imputer + one-hot encoder
  - final estimator from `models.py`

- `models.py` — model factory:
  - standard models via `family/kind/params`
  - complex models via `builder` functions (string path like `"builders.houseprices_stacking"`)

- `builders.py` — complex model constructors:
  - example: `houseprices_stacking()` returns a `StackingRegressor`

- `trainer.py` — training logic:
  - CV loop (fold metrics)
  - aggregates mean/std
  - selects best model
  - trains best model on full train
  - writes `best_report.json`

- `metrics.py` — fold-level metrics:
  - classification: accuracy / f1 / roc_auc
  - regression: rmse / mae / r2

- `target_transformer.py` — target transform for regression:
  - train/CV: `log1p`
  - submission: inverse `expm1`

- `selection.py` — model selection policy:
  - `mean`
  - `std`
  - `mean_minus_k_std`

- `predict.py` — loads saved model and returns predictions

---

## Model selection

Selection is controlled by:

- `TRAIN_CONFIG["metric"]` — which metric to optimize (e.g. `"accuracy"`, `"rmse"`)
- `TRAIN_CONFIG["direction"]` — `"max"` or `"min"`
- `TRAIN_CONFIG["selection"]` — how to compute the score used for ranking models:
  - `{"method": "mean"}` — uses CV mean of the metric
  - `{"method": "std"}` — uses CV std (stability selection; usually `direction="min"`)
  - `{"method": "mean_minus_k_std", "k": 1.0}` — penalizes instability

---

## How to add a model

### A) Standard model
1. Add a new entry to `MODEL_CANDIDATES` in `config.py`
2. Ensure `models.py` supports the `family`

Example (ExtraTrees):
```python
MODEL_CANDIDATES["extra_trees"] = {
    "family": "extra_trees",
    "kind": "classifier",
    "requires_scaling": False,
    "params": {"n_estimators": 800, "random_state": 42, "n_jobs": -1},
}
SELECTED_MODELS.append("extra_trees")
```

Then add a branch in `models.py` for `"extra_trees"`.

### B) Builder model 
1. Implement a function in `builders.py`
2. Reference it in `MODEL_CANDIDATES` using `"builder": "builders.func_name"`

Example:
```python
MODEL_CANDIDATES["stacking_hp"] = {
    "builder": "builders.houseprices_stacking",
    "kind": "regressor",
    "requires_scaling": False,
    "params": {
        "cv_splits": 5,
        "cb_params": {"iterations": 4000, "depth": 8},
    },
}
SELECTED_MODELS.append("stacking_hp")
```

---

## Results (Cross-Validation)

### Titanic (Classification)
**Target:** `Survived`  
**Submission format:** `PassengerId,Survived`  
**Primary metric (CV):** Accuracy 

**Best model:** `rf_classifier`  
**Best CV Accuracy:** **0.8294 ± 0.0182**

**Model comparison (selection score):**
- `rf_classifier` — 0.8112
- `cat_classifier` — 0.8064
- `logreg` — 0.7698
- `xgb_classifier` — 0.7572

### House Prices (Regression)
**Target:** `SalePrice`  
**Submission format:** `Id,SalePrice`  
**Target transform:** `log1p` (train/CV) 
**Primary metric (CV):** RMSE on `log1p(SalePrice)` 

**Best model:** `cat_regressor`  
**Best CV RMSE (log1p):** **0.1228 ± 0.0198**

**Leaderboard (CV RMSE, log1p):**
- `cat_regressor` — **0.1228**
- `stacking_hp` — 0.1278
- `xgb_regressor` — 0.1339
- `ridge` — 0.1485
