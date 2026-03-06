from __future__ import annotations
import pandas as pd
import joblib

def make_predictions(model_path: str, X_test: pd.DataFrame):
    model = joblib.load(model_path)
    return model.predict(X_test)
