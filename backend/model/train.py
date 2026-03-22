import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

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

# Regressor params — same tree structure, different eval metric
XGB_REG_FIXED = {
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": -1,
}


def _get_model_path(ticker: str) -> Path:
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return MODEL_DIR / f"{symbol}_xgb.joblib"


def _get_reg_model_path(ticker: str) -> Path:
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return MODEL_DIR / f"{symbol}_xgb_reg.joblib"


def train_model(
    ticker: str,
    feature_df: pd.DataFrame,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """
    Train an XGBoost classifier + regressor using walk-forward validation,
    then train an LSTM on the full dataset.

    Walk-forward strategy:
        - Expanding window starting at WALK_FORWARD_WINDOW (252 days)
        - Each fold: train on all prior data, validate on next 63 days (1 quarter)
        - Final models trained on ALL available data for live prediction

    Returns: (classifier_model, training_metadata_dict)
    The regressor and LSTM are saved to separate files.
    """
    feature_cols = get_feature_columns(feature_df)

    df = feature_df.copy()
    df[feature_cols] = df[feature_cols].ffill().fillna(0.0)

    # Drop rows where neither target is known
    df = df.dropna(subset=["target"])

    # Load best Optuna params if tuning has been run for this ticker.
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

    # Regressor uses same tree params but MAE eval metric
    xgb_reg_params = {
        k: v for k, v in xgb_params.items()
        if k not in XGB_PARAMS_FIXED
    }
    xgb_reg_params.update(XGB_REG_FIXED)

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

        # Regressor
        df_reg = df.dropna(subset=["target_return"])
        if len(df_reg) > 10:
            y_ret = df_reg["target_return"].astype(float)
            regressor = XGBRegressor(**xgb_reg_params)
            regressor.fit(df_reg[feature_cols], y_ret)
            _save_regressor(ticker, regressor, feature_cols)

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
    oos_records: list[dict] = []

    for fold_start in range(WALK_FORWARD_WINDOW, len(df) - fold_size, fold_size):
        train_slice = df.iloc[:fold_start]
        val_slice   = df.iloc[fold_start : fold_start + fold_size]

        X_train = train_slice[feature_cols]
        y_train = train_slice["target"].astype(int)
        X_val   = val_slice[feature_cols]
        y_val   = val_slice["target"].astype(int)

        fold_model = XGBClassifier(**xgb_params)
        fold_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds  = fold_model.predict(X_val)
        proba  = fold_model.predict_proba(X_val)
        classes = list(fold_model.classes_)
        up_idx  = classes.index(1) if 1 in classes else 1

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
            up_prob  = float(proba[j, up_idx])
            conf     = up_prob if pred_dir == 1 else 1 - up_prob
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

    mean_accuracy = (
        sum(f["accuracy"] for f in fold_results) / len(fold_results)
        if fold_results else 0.5
    )
    logger.info(
        "Walk-forward CV mean accuracy for %s: %.4f (%d folds)",
        ticker, mean_accuracy, len(fold_results),
    )

    # ── Final classifier: train on ALL data ───────────────────────────────────
    X_full = df[feature_cols]
    y_full = df["target"].astype(int)
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X_full, y_full, verbose=False)

    # ── Final regressor: train on ALL data ────────────────────────────────────
    df_reg = df.dropna(subset=["target_return"])
    if len(df_reg) > 30:
        y_ret = df_reg["target_return"].astype(float)
        regressor = XGBRegressor(**xgb_reg_params)
        regressor.fit(df_reg[feature_cols], y_ret, verbose=False)
        _save_regressor(ticker, regressor, feature_cols)
        logger.info("Trained price regressor for %s (%d rows)", ticker, len(df_reg))
    else:
        logger.warning("Not enough rows for regressor training for %s", ticker)

    # ── LSTM: train on ALL data (adds ~30s, runs after main pipeline) ─────────
    try:
        from model.lstm_model import train_lstm
        lstm_meta = train_lstm(ticker, feature_df, feature_cols)
        logger.info(
            "LSTM trained for %s — walk-forward accuracy: %.4f",
            ticker, lstm_meta.get("accuracy", 0),
        )
    except Exception as exc:
        logger.warning("LSTM training failed for %s (non-fatal): %s", ticker, exc)

    # ── Feature importances ───────────────────────────────────────────────────
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features for %s: %s", ticker, top_features)

    # ── Save classifier ───────────────────────────────────────────────────────
    model_path = _get_model_path(ticker)
    oos_df = (
        pd.DataFrame(oos_records).sort_values("date").reset_index(drop=True)
        if oos_records else pd.DataFrame()
    )

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
        "xgb_params_used": xgb_params,
        "used_tuned_params": bool(tuned),
    }
    joblib.dump({"model": final_model, "metadata": metadata}, model_path)
    logger.info("Saved model for %s to %s", ticker, model_path)

    return final_model, metadata


def _save_regressor(
    ticker: str,
    regressor: XGBRegressor,
    feature_cols: list[str],
) -> None:
    path = _get_reg_model_path(ticker)
    joblib.dump({"regressor": regressor, "feature_cols": feature_cols}, path)


def load_regressor(ticker: str) -> tuple[XGBRegressor, list[str]] | tuple[None, None]:
    """Load the price regressor. Returns (None, None) if not yet trained."""
    path = _get_reg_model_path(ticker)
    if not path.exists():
        return None, None
    saved = joblib.load(path)
    return saved["regressor"], saved["feature_cols"]


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
