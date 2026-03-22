import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier

from config import MODEL_DIR, WALK_FORWARD_WINDOW
from features.feature_builder import get_feature_columns

# Import lazily to avoid circular import — tune.py also imports train.py
def _get_best_params(ticker: str) -> dict:
    try:
        from model.tune import get_best_params
        return get_best_params(ticker) or {}
    except Exception:
        return {}

logger = logging.getLogger(__name__)

XGB_PARAMS_DEFAULT = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.0,
}

# These are always applied on top of tuned params — never overridden
XGB_PARAMS_FIXED = {
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def _get_model_path(ticker: str) -> Path:
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return MODEL_DIR / f"{symbol}_xgb.joblib"


def train_model(
    ticker: str,
    feature_df: pd.DataFrame,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """
    Train an XGBoost classifier using walk-forward validation.

    Walk-forward strategy:
        - Expanding window starting at WALK_FORWARD_WINDOW (252 days)
        - Each fold: train on all prior data, validate on next 63 days (1 quarter)
        - Final model trained on ALL available data for live prediction

    Returns: (trained_model, training_metadata_dict)
    """
    feature_cols = get_feature_columns(feature_df)

    # Fill remaining NaNs in features before training — using forward-fill then
    # zero-fill so no rows are lost due to a single missing indicator value.
    df = feature_df.copy()
    df[feature_cols] = df[feature_cols].ffill().fillna(0.0)

    # Only drop rows where the TARGET is unknown (last row + any edge cases)
    df = df.dropna(subset=["target"])

    # Load best Optuna params if tuning has been run for this ticker.
    # Falls back to defaults on first run (before any tuning).
    tuned = _get_best_params(ticker)
    xgb_params = {**XGB_PARAMS_DEFAULT, **tuned, **XGB_PARAMS_FIXED}
    if tuned:
        logger.info(
            "Using Optuna-tuned params for %s: n_estimators=%s, max_depth=%s, lr=%.4f",
            ticker,
            tuned.get("n_estimators", "default"),
            tuned.get("max_depth", "default"),
            tuned.get("learning_rate", XGB_PARAMS_DEFAULT["learning_rate"]),
        )
    else:
        logger.info("Using default XGB params for %s (no tuning done yet)", ticker)

    if len(df) < WALK_FORWARD_WINDOW + 63:
        logger.warning(
            "Insufficient data for walk-forward validation (%d rows). "
            "Training on all available data.",
            len(df),
        )
        X = df[feature_cols]
        y = df["target"].astype(int)
        model = XGBClassifier(**xgb_params)
        model.fit(X, y)
        model_path = _get_model_path(ticker)
        metadata = {
            "ticker": ticker,
            "feature_cols": feature_cols,
            "accuracy": 0.5,
            "n_folds": 0,
            "trained_on_rows": len(df),
        }
        joblib.dump({"model": model, "metadata": metadata}, model_path)
        return model, metadata

    # ── Walk-forward validation ───────────────────────────────────────────────
    fold_results = []
    fold_size = 63  # ~1 quarter
    oos_records: list[dict] = []  # out-of-sample predictions for honest backtest

    for fold_start in range(WALK_FORWARD_WINDOW, len(df) - fold_size, fold_size):
        train_slice = df.iloc[:fold_start]
        val_slice = df.iloc[fold_start : fold_start + fold_size]

        X_train = train_slice[feature_cols]
        y_train = train_slice["target"].astype(int)
        X_val = val_slice[feature_cols]
        y_val = val_slice["target"].astype(int)

        fold_model = XGBClassifier(**xgb_params)
        fold_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = fold_model.predict(X_val)
        proba = fold_model.predict_proba(X_val)
        classes = list(fold_model.classes_)
        up_idx = classes.index(1) if 1 in classes else 1

        accuracy = (preds == y_val.values).mean()
        fold_results.append(
            {
                "fold_start": df.index[fold_start],
                "accuracy": accuracy,
                "n_train": len(train_slice),
            }
        )

        # Store out-of-sample predictions for honest backtest
        for j, (idx, row) in enumerate(val_slice.iterrows()):
            pred_dir = int(preds[j])
            up_prob = float(proba[j, up_idx])
            conf = up_prob if pred_dir == 1 else 1 - up_prob
            oos_records.append({
                "date": idx,
                "actual_price": float(row["Close"]),
                "predicted_direction": pred_dir,
                "prediction_confidence": conf,
            })

        logger.debug(
            "Fold %d/%d — train_size=%d, accuracy=%.4f",
            len(fold_results),
            (len(df) - WALK_FORWARD_WINDOW) // fold_size,
            len(train_slice),
            accuracy,
        )

    mean_accuracy = sum(f["accuracy"] for f in fold_results) / len(fold_results) if fold_results else 0.5
    logger.info("Walk-forward CV mean accuracy for %s: %.4f (%d folds)", ticker, mean_accuracy, len(fold_results))

    # ── Final model: train on ALL data ───────────────────────────────────────
    X_full = df[feature_cols]
    y_full = df["target"].astype(int)
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X_full, y_full, verbose=False)

    # ── Feature importances ───────────────────────────────────────────────────
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features for %s: %s", ticker, top_features)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = _get_model_path(ticker)
    # Build out-of-sample DataFrame for honest backtest
    oos_df = pd.DataFrame(oos_records).sort_values("date").reset_index(drop=True) if oos_records else pd.DataFrame()

    metadata = {
        "ticker": ticker,
        "feature_cols": feature_cols,
        "accuracy": mean_accuracy,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "feature_importances": importances,
        "top_features": top_features,
        "trained_on_rows": len(df),
        "oos_predictions": oos_df,
        "xgb_params_used": xgb_params,   # which params actually trained this model
        "used_tuned_params": bool(tuned), # whether Optuna params were applied
    }
    joblib.dump({"model": final_model, "metadata": metadata}, model_path)
    logger.info("Saved model for %s to %s", ticker, model_path)

    return final_model, metadata


def load_model(ticker: str) -> tuple[XGBClassifier, dict[str, Any]]:
    """Load a previously trained model from disk."""
    model_path = _get_model_path(ticker)
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {ticker} at {model_path}")
    saved = joblib.load(model_path)
    return saved["model"], saved["metadata"]


def model_exists(ticker: str) -> bool:
    """Check whether a trained model exists for a ticker."""
    return _get_model_path(ticker).exists()
