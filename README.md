# Universal tabular ML template (single `config.py`) — v3 (builders)

Adds **builder models** (e.g. StackingRegressor) without hardcoding per-dataset logic in the core.

## Key ideas
- Most models use: `family/kind/params`
- Complex models can use: `builder` (a function that returns an sklearn estimator)

Example in `config.py`:
```py
MODEL_CANDIDATES["stacking_hp"] = {
    "builder": "builders.houseprices_stacking",
    "kind": "regressor",
    "requires_scaling": False,
    "params": {}
}
```

## Run
```bash
pip install -r requirements.txt
python main.py
```

## House Prices quick setup
In `config.py`:
- DATA_CONFIG["target"] = "SalePrice"
- DATA_CONFIG["id_col"] = "Id"
- TRAIN_CONFIG["task"] = "regression"
- TRAIN_CONFIG["metric"] = "rmse"
- TRAIN_CONFIG["direction"] = "min"
- TRAIN_CONFIG["target_transform"] = "log1p"
- TRAIN_CONFIG["prediction_column"] = "SalePrice"
- SELECTED_MODELS includes: "stacking_hp"
