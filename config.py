from __future__ import annotations

DATA_CONFIG = {
    "train_path": "data/raw/train.csv",
    "test_path": "data/raw/test.csv",  
    "target": "SalePrice",
    "id_col": "Id",                   
    "drop_cols": [],
}

TRAIN_CONFIG = {
    "task": "regression",         # "classification" or "regression"
    "metric": "rmse",              # classification: accuracy/f1/roc_auc ; regression: rmse/mae/r2
    "direction": "min",               # "max" or "min"
    "cv": 5,
    "shuffle": True,
    "seed": 42,
    "save_model": True,
    "save_report": True,
    "submission_filename": "submission.csv",
    "prediction_column": "SalePrice",
    "outdir": "artifacts",

    # Regression-only target transform: None | "log1p"
    "target_transform": "log1p",
    "selection": {"method": "mean_minus_k_std", "k": 1.0},
}

MODEL_CANDIDATES = {
    # classification
    "logreg": {
        "family": "logreg",
        "kind": "classifier",
        "requires_scaling": True,
        "params": {"C": 1.0, "max_iter": 5000, "solver": "lbfgs", "verbose": False,},
    },
    "rf_classifier": {
        "family": "rf",
        "kind": "classifier",
        "requires_scaling": False,
        "params": {"n_estimators": 500, "random_state": 42, "max_depth": None, "n_jobs": -1, "verbose": False,},
    },
    "xgb_classifier": {
        "family": "xgb",
        "kind": "classifier",
        "requires_scaling": False,
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "verbose": False,
        },
    },
    "cat_classifier": {
        "family": "catboost",
        "kind": "classifier",
        "requires_scaling": False,
        "params": {
            "iterations": 2500,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": False,
        },
    },

    # regression
    "ridge": {
        "family": "ridge",
        "kind": "regressor",
        "requires_scaling": True,
        "params": {"alpha": 1.0},
    },
    "rf_regressor": {
        "family": "rf",
        "kind": "regressor",
        "requires_scaling": False,
        "params": {"n_estimators": 600, "random_state": 42, "max_depth": None, "n_jobs": -1},
    },
    "xgb_regressor": {
        "family": "xgb",
        "kind": "regressor",
        "requires_scaling": False,
        "params": {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
        },
    },
    "cat_regressor": {
        "family": "catboost",
        "kind": "regressor",
        "requires_scaling": False,
        "params": {
            "iterations": 3500,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
        },
    },

    "stacking_hp": {
        "builder": "builders.houseprices_stacking",
        "kind": "regressor",
        "requires_scaling": False,
        "params": {"verbose": False},
    },
}

SELECTED_MODELS = ["stacking_hp", "cat_regressor", "xgb_regressor", "ridge"]