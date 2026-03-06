from __future__ import annotations
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add dataset-specific feature engineering here (must work for both train and test).
    return df.copy()
