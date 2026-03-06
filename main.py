from __future__ import annotations
from pathlib import Path
import pandas as pd

from config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CANDIDATES, SELECTED_MODELS
from data import read_csv, get_xy, drop_cols_safe
from features import build_features
from trainer import fit_best_model
from predict import make_predictions
from target_transformer import TargetTransformer

def main():
    train_path = Path(DATA_CONFIG["train_path"])
    test_path = Path(DATA_CONFIG.get("test_path", ""))
    target = DATA_CONFIG["target"]
    id_col = DATA_CONFIG.get("id_col")
    drop_cols = DATA_CONFIG.get("drop_cols", [])
    outdir = TRAIN_CONFIG["outdir"]

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df_train = build_features(read_csv(train_path))
    X, y = get_xy(df_train, target=target, drop_cols=drop_cols)

    result = fit_best_model(
        X=X,
        y=y,
        model_candidates=MODEL_CANDIDATES,
        selected_models=SELECTED_MODELS,
        train_cfg=TRAIN_CONFIG,
        outdir=outdir,
    )

    print(f"Best model: {result['best_model']}")
    print(f"Best selection score: {result['best_score']:.6f}")
    print(f"Model saved to: {result['model_path']}")
    print(f"Report saved to: {result['report_path']}")

    if test_path and test_path.exists():
        df_test = build_features(read_csv(test_path))
        X_test = drop_cols_safe(df_test, drop_cols)

        preds = make_predictions(result["model_path"], X_test)

        transformer = TargetTransformer(TRAIN_CONFIG.get("target_transform"))
        if TRAIN_CONFIG.get("task") == "regression":
            preds = transformer.inverse(preds)

        pred_col = TRAIN_CONFIG.get("prediction_column", "prediction")
        submission = pd.DataFrame({pred_col: preds})

        if id_col and id_col in df_test.columns:
            submission.insert(0, id_col, df_test[id_col].values)

        sub_path = Path(outdir) / "submissions" / TRAIN_CONFIG.get("submission_filename", "submission.csv")
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to: {sub_path}")

if __name__ == "__main__":
    main()
