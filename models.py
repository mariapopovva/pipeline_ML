from __future__ import annotations
from typing import Any, Dict
import importlib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

def _load_builder(builder_path: str):
    mod_name, fn_name = builder_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn

def build_model(model_cfg: Dict[str, Any]) -> Any:
    builder_path = model_cfg.get("builder")
    if builder_path:
        fn = _load_builder(builder_path)
        params = dict(model_cfg.get("params", {}) or {})
        return fn(params)

    family = model_cfg.get("family")
    kind = model_cfg.get("kind")
    params = dict(model_cfg.get("params", {}) or {})

    if family == "rf":
        return RandomForestClassifier(**params) if kind == "classifier" else RandomForestRegressor(**params)

    if family == "logreg":
        return LogisticRegression(**params)

    if family == "ridge":
        return Ridge(**params)

    if family == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        if kind == "classifier":
            eval_metric = params.pop("eval_metric", "logloss")
            return XGBClassifier(**params, eval_metric=eval_metric)
        return XGBRegressor(**params)

    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier(**params) if kind == "classifier" else CatBoostRegressor(**params)

    raise ValueError(f"Unknown model family: {family}")
