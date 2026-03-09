# Universal ML template

## Run
pip install -r requirements.txt <br>
python main.py


## House Prices quick setup
In `config.py`:<br>
DATA_CONFIG["target"] = "SalePrice" <br>
DATA_CONFIG["id_col"] = "Id"<br>

TRAIN_CONFIG["task"] = "regression"<br>
TRAIN_CONFIG["metric"] = "rmse"<br>
TRAIN_CONFIG["direction"] = "min"<br>
TRAIN_CONFIG["target_transform"] = "log1p"<br>
TRAIN_CONFIG["prediction_column"] = "SalePrice"<br>


## Titanic
In `config.py`:<br>
DATA_CONFIG["target"] = "Survived"<br>
DATA_CONFIG["id_col"] = "PassengerId"<br>

TRAIN_CONFIG["task"] = "classification"<br>
TRAIN_CONFIG["metric"] = "accuracy"<br>
TRAIN_CONFIG["direction"] = "max"<br>
TRAIN_CONFIG["target_transform"] = None<br>
TRAIN_CONFIG["prediction_column"] = "Survived"<br>

## Results (Cross-Validation)

### Titanic (Binary Classification)
**Target:** `Survived`  
**Submission format:** `PassengerId,Survived`  

**Primary metric (CV):** Accuracy (higher is better)  
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
**Target transform:** `log1p` during training, `expm1` for submission  

**Primary metric (CV):** RMSE on `log1p(SalePrice)` (lower is better)  
**Best model:** `cat_regressor`  
**Best CV RMSE (log1p):** **0.1228 ± 0.0198**

**Leaderboard (CV RMSE, log1p):**
- `cat_regressor` — **0.1228**
- `stacking_hp` — 0.1278
- `xgb_regressor` — 0.1339
- `ridge` — 0.1485
