from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import build_model

def infer_column_roles(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_pipeline(X: pd.DataFrame, model_cfg: dict) -> Pipeline:
    numeric_cols, categorical_cols = infer_column_roles(X)
    scale_numeric = bool(model_cfg.get("requires_scaling", False))

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric and numeric_cols:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = build_model(model_cfg)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
