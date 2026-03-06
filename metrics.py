from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def compute_metrics(task: str, metrics: List[str], y_true, y_pred, y_proba=None) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if task == "classification":
        n_classes = len(np.unique(y_true))
        for metric in metrics:
            if metric == "accuracy":
                out[metric] = float(accuracy_score(y_true, y_pred))
            elif metric == "f1":
                avg = "binary" if n_classes == 2 else "weighted"
                out[metric] = float(f1_score(y_true, y_pred, average=avg))
            elif metric == "roc_auc":
                if y_proba is None:
                    continue
                if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2:
                    out[metric] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    out[metric] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return out

    if task == "regression":
        for metric in metrics:
            if metric == "rmse":
                mse = mean_squared_error(y_true, y_pred)
                out[metric] = float(np.sqrt(mse))
            elif metric == "mae":
                out[metric] = float(mean_absolute_error(y_true, y_pred))
            elif metric == "r2":
                out[metric] = float(r2_score(y_true, y_pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return out

    raise ValueError(f"Unknown task: {task}")

def default_metrics_for_task(task: str) -> list[str]:
    if task == "classification":
        return ["accuracy", "f1", "roc_auc"]
    if task == "regression":
        return ["rmse", "mae", "r2"]
    raise ValueError(f"Unknown task: {task}")
