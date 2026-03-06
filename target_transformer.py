from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class TargetTransformer:
    name: Optional[str] = None  # None | "log1p"

    def transform(self, y: pd.Series) -> pd.Series:
        if self.name is None or self.name == "none":
            return y
        if self.name == "log1p":
            return np.log1p(y)
        raise ValueError(f"Unknown target_transform: {self.name}")

    def inverse(self, y_pred):
        if self.name is None or self.name == "none":
            return y_pred
        if self.name == "log1p":
            return np.expm1(y_pred)
        raise ValueError(f"Unknown target_transform: {self.name}")
