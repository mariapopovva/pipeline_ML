from __future__ import annotations
from typing import Dict, Any

def selection_score(cv_report: Dict[str, Any], metric: str, selection: Dict[str, Any]) -> float:
    method = (selection or {}).get("method", "mean")
    k = float((selection or {}).get("k", 1.0))

    mean = float(cv_report["mean"][metric])
    std = float(cv_report["std"][metric])

    if method == "mean":
        return mean
    if method == "std":
        return std
    if method == "mean_minus_k_std":
        return mean - k * std
    raise ValueError("selection.method must be: mean | std | mean_minus_k_std")
