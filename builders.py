from __future__ import annotations
from typing import Any, Dict
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

def houseprices_stacking(params: Dict[str, Any] | None = None):
    params = params or {}

    cb_params = {
        "loss_function": "RMSE",
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "random_seed": 42,
        "verbose": False,
    }
    cb_params.update(params.get("cb_params", {}))

    ridge_params = {"alpha": 20, "solver": "svd"}
    ridge_params.update(params.get("ridge_params", {}))

    gbr_params = {"random_state": 42}
    gbr_params.update(params.get("gbr_params", {}))

    final_alphas = params.get("final_alphas", [0.1, 1, 10, 100])
    cv_splits = int(params.get("cv_splits", 5))
    cv_seed = int(params.get("cv_seed", 42))

    estimators = [
        ("cb", CatBoostRegressor(**cb_params)),
        ("r", Ridge(**ridge_params)),
        ("gb", GradientBoostingRegressor(**gbr_params)),
    ]

    final_estimator = RidgeCV(alphas=final_alphas)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)

    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1,
        cv=kf,
    )
