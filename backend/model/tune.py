"""
Hyperparameter tuning for XGBoost models using Optuna.

How it works:
  - Each ticker gets its own persistent Optuna study stored on disk.
  - Every time a stock is refreshed, 30 more trials are added to that study.
  - Optuna's TPE sampler learns from all past trials — it gets smarter over time,
    focusing search on parameter regions that gave the best walk-forward accuracy.
  - The best params found so far are saved and used by train.py on the next run.
  - Result: the model genuinely improves with each Refresh, converging on the
    optimal hyperparameters for each specific stock's price behaviour.

Optuna study files are stored at:
    backend/model/saved/<TICKER>_optuna_study.pkl

To reset a stock's tuning history (start fresh):
    delete the .pkl file and re-add the stock.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from xgboost import XGBClassifier

from config import MODEL_DIR, WALK_FORWARD_WINDOW
from features.feature_builder import get_feature_columns

logger = logging.getLogger(__name__)

# Silence Optuna's own verbose logging — we log our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Tuning budget per refresh:
#   First 60 trials — exploration (broad search)
#   After 60 trials  — exploitation (TPE focuses on best regions, more trials)
N_TRIALS_BASE  = 35    # per refresh before we have history
N_TRIALS_EXPLOIT = 60  # per refresh once we have 60+ past trials
FOLD_SIZE = 63      # ~1 quarter, same as train.py


def _study_path(ticker: str) -> Path:
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return MODEL_DIR / f"{symbol}_optuna_study.pkl"


def _walk_forward_score(
    params: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """
    Run walk-forward CV with given params and return mean accuracy.
    Same logic as train.py so results are directly comparable.
    """
    scores = []
    for fold_start in range(WALK_FORWARD_WINDOW, len(df) - FOLD_SIZE, FOLD_SIZE):
        train_slice = df.iloc[:fold_start]
        val_slice   = df.iloc[fold_start : fold_start + FOLD_SIZE]

        X_train = train_slice[feature_cols]
        y_train = train_slice["target"].astype(int)
        X_val   = val_slice[feature_cols]
        y_val   = val_slice["target"].astype(int)

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        scores.append((preds == y_val.values).mean())

    return sum(scores) / len(scores) if scores else 0.5


def tune_hyperparameters(
    ticker: str,
    feature_df: pd.DataFrame,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search for a ticker.

    Loads the existing study (if any) so Optuna builds on previous knowledge,
    runs `n_trials` more trials, saves the updated study, and returns the
    best params found across ALL trials ever run for this ticker.

    Called as a background task after the main pipeline — never blocks the UI.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    study_path = _study_path(symbol)

    # ── Prepare feature matrix (same NaN handling as train.py) ───────────────
    feature_cols = get_feature_columns(feature_df)
    df = feature_df.copy()
    df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
    df = df.dropna(subset=["target"])

    if len(df) < WALK_FORWARD_WINDOW + FOLD_SIZE:
        logger.warning(
            "Tuning skipped for %s — not enough data (%d rows)", symbol, len(df)
        )
        return {}

    # ── Load or create study ──────────────────────────────────────────────────
    if study_path.exists():
        try:
            study = joblib.load(study_path)
            past_trials = len(study.trials)
            logger.info(
                "Loaded Optuna study for %s (%d past trials)", symbol, past_trials
            )
        except Exception as exc:
            logger.warning("Could not load Optuna study for %s: %s — starting fresh", symbol, exc)
            study = optuna.create_study(direction="maximize", study_name=symbol)
            past_trials = 0
    else:
        study = optuna.create_study(direction="maximize", study_name=symbol)
        past_trials = 0

    # More trials once we have a history to exploit
    if n_trials is None:
        n_trials = N_TRIALS_EXPLOIT if past_trials >= 60 else N_TRIALS_BASE

    # ── Define search space ───────────────────────────────────────────────────
    # 16 hyperparameters covering tree structure, regularisation, sampling,
    # and boosting schedule.  Wider ranges allow Optuna to discover unusual
    # but effective configs that hand-tuning would miss.
    def objective(trial: optuna.Trial) -> float:
        # ── Boosting schedule ──────────────────────────────────────────────
        n_estimators  = trial.suggest_int("n_estimators", 80, 800)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.4, log=True)

        # ── Tree structure ─────────────────────────────────────────────────
        max_depth          = trial.suggest_int("max_depth", 2, 10)
        min_child_weight   = trial.suggest_int("min_child_weight", 1, 50)
        max_delta_step     = trial.suggest_int("max_delta_step", 0, 10)
        gamma              = trial.suggest_float("gamma", 0.0, 5.0)

        # ── Regularisation ─────────────────────────────────────────────────
        reg_alpha  = trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

        # ── Feature & row sampling ─────────────────────────────────────────
        subsample          = trial.suggest_float("subsample", 0.4, 1.0)
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        colsample_bylevel  = trial.suggest_float("colsample_bylevel", 0.3, 1.0)
        colsample_bynode   = trial.suggest_float("colsample_bynode", 0.3, 1.0)

        # ── Tree growing policy ────────────────────────────────────────────
        # "depthwise" is standard; "lossguide" is LightGBM-style leaf-wise
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        # max_leaves only applies to lossguide
        max_leaves  = trial.suggest_int("max_leaves", 0, 256) if grow_policy == "lossguide" else 0

        # ── Class imbalance correction ─────────────────────────────────────
        # Most NSE stocks have roughly equal up/down days, but this lets
        # Optuna discover if slight weighting helps.
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 2.0)

        params = {
            "n_estimators":       n_estimators,
            "learning_rate":      learning_rate,
            "max_depth":          max_depth,
            "min_child_weight":   min_child_weight,
            "max_delta_step":     max_delta_step,
            "gamma":              gamma,
            "reg_alpha":          reg_alpha,
            "reg_lambda":         reg_lambda,
            "subsample":          subsample,
            "colsample_bytree":   colsample_bytree,
            "colsample_bylevel":  colsample_bylevel,
            "colsample_bynode":   colsample_bynode,
            "grow_policy":        grow_policy,
            "max_leaves":         max_leaves,
            "scale_pos_weight":   scale_pos_weight,
        }
        return _walk_forward_score(params, df, feature_cols)

    # ── Run trials ────────────────────────────────────────────────────────────
    logger.info(
        "Tuning %s: running %d Optuna trials (total after: %d)",
        symbol, n_trials, past_trials + n_trials,
    )
    study.optimize(objective, n_trials=n_trials, timeout=180, show_progress_bar=False)

    best = study.best_params
    best_score = study.best_value

    logger.info(
        "Tuning complete for %s — best walk-forward accuracy: %.4f after %d total trials",
        symbol, best_score, len(study.trials),
    )
    logger.info("Best params for %s: %s", symbol, best)

    # ── Save updated study ────────────────────────────────────────────────────
    try:
        joblib.dump(study, study_path)
    except Exception as exc:
        logger.warning("Could not save Optuna study for %s: %s", symbol, exc)

    return best


def get_best_params(ticker: str) -> dict[str, Any] | None:
    """
    Return the best hyperparameters found so far for a ticker.
    Returns None if no tuning has been done yet.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    path = _study_path(symbol)
    if not path.exists():
        return None
    try:
        study = joblib.load(path)
        return study.best_params if study.trials else None
    except Exception:
        return None


def get_tuning_history(ticker: str) -> dict[str, Any]:
    """
    Return a summary of tuning history for a ticker (for the frontend stats page).
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    path = _study_path(symbol)
    if not path.exists():
        return {"tuned": False, "n_trials": 0, "best_accuracy": None}
    try:
        study = joblib.load(path)
        return {
            "tuned": True,
            "n_trials": len(study.trials),
            "best_accuracy": round(study.best_value * 100, 2) if study.trials else None,
            "best_params": study.best_params if study.trials else None,
        }
    except Exception:
        return {"tuned": False, "n_trials": 0, "best_accuracy": None}
