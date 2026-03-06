from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))

def drop_cols_safe(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    drop_cols = drop_cols or []
    present = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=present)

def get_xy(df: pd.DataFrame, target: str, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    X = df.drop(columns=[target])
    X = drop_cols_safe(X, drop_cols)
    y = df[target]
    return X, y
