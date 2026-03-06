from __future__ import annotations
from pathlib import Path
import json
import joblib
from typing import Dict, Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from pipeline import build_pipeline
from metrics import compute_metrics, default_metrics_for_task
from target_transformer import TargetTransformer
from selection import selection_score

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_cv(task: str, n_splits: int, shuffle: bool, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

def evaluate_single_model(X: pd.DataFrame, y: pd.Series, model_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    task = train_cfg["task"]
    metric_list = default_metrics_for_task(task)
    cv = make_cv(task, train_cfg["cv"], train_cfg["shuffle"], train_cfg["seed"])

    transformer = TargetTransformer(train_cfg.get("target_transform"))
    y_used = transformer.transform(y) if task == "regression" else y

    pipe = build_pipeline(X, model_cfg)
    fold_results = []

    splitter = cv.split(X, y_used if task == "classification" else None)
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter, start=1):
        model_i = clone(pipe)

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_used.iloc[tr_idx], y_used.iloc[va_idx]

        model_i.fit(X_tr, y_tr)
        y_pred = model_i.predict(X_va)

        y_proba = None
        if task == "classification" and hasattr(model_i, "predict_proba"):
            try:
                y_proba = model_i.predict_proba(X_va)
            except Exception:
                y_proba = None

        metrics = compute_metrics(task, metric_list, y_va, y_pred, y_proba=y_proba)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)
    mean_metrics = fold_df.drop(columns=["fold"]).mean().to_dict()
    std_metrics = fold_df.drop(columns=["fold"]).std().fillna(0.0).to_dict()

    return {
        "folds": fold_results,
        "mean": {k: float(v) for k, v in mean_metrics.items()},
        "std": {k: float(v) for k, v in std_metrics.items()},
    }

def fit_best_model(X: pd.DataFrame, y: pd.Series, model_candidates: Dict[str, Any], selected_models: list[str], train_cfg: Dict[str, Any], outdir: str):
    ensure_dir(outdir)
    ensure_dir(Path(outdir) / "models")
    ensure_dir(Path(outdir) / "reports")
    ensure_dir(Path(outdir) / "submissions")

    leaderboard = []
    score_metric = train_cfg["metric"]
    sel = train_cfg.get("selection", {}) or {}

    for model_name in selected_models:
        if model_name not in model_candidates:
            raise ValueError(f"Model '{model_name}' not found in MODEL_CANDIDATES")

        model_cfg = model_candidates[model_name]
        cv_report = evaluate_single_model(X, y, model_cfg, train_cfg)

        if score_metric not in cv_report["mean"]:
            raise ValueError(
                f"Metric '{score_metric}' not available for model '{model_name}'. "
                f"Available: {list(cv_report['mean'].keys())}"
            )

        score = selection_score(cv_report, score_metric, sel)
        leaderboard.append({"model": model_name, "score": float(score), "cv_report": cv_report})

    reverse = train_cfg["direction"] == "max"
    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=reverse)
    best = leaderboard[0]

    best_name = best["model"]
    best_cfg = model_candidates[best_name]
    best_pipeline = build_pipeline(X, best_cfg)

    transformer = TargetTransformer(train_cfg.get("target_transform"))
    y_fit = transformer.transform(y) if train_cfg["task"] == "regression" else y
    best_pipeline.fit(X, y_fit)

    model_path = Path(outdir) / "models" / "model.joblib"
    if train_cfg.get("save_model", True):
        joblib.dump(best_pipeline, model_path)

    report = {
        "task": train_cfg["task"],
        "metric": train_cfg["metric"],
        "direction": train_cfg["direction"],
        "selection": train_cfg.get("selection", {}),
        "target_transform": train_cfg.get("target_transform"),
        "best_model": best_name,
        "best_score": best["score"],
        "leaderboard": [{"model": row["model"], "score": row["score"]} for row in leaderboard],
        "best_cv": best["cv_report"],
    }

    report_path = Path(outdir) / "reports" / "best_report.json"
    if train_cfg.get("save_report", True):
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "best_model": best_name,
        "best_score": best["score"],
        "model_path": str(model_path),
        "report_path": str(report_path),
        "leaderboard": leaderboard,
    }
