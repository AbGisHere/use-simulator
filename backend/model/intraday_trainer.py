"""
Intraday 10-minute model for NSE stocks.

This is a lightweight, fast-training model designed to:
  1. Retrain every 10 minutes on the latest intraday bars (~30ms)
  2. Predict the next 10-minute price direction + magnitude
  3. Track running prediction accuracy during the session

Architecture:
  - XGBoost Classifier  → next 10-min direction (up/down)
  - XGBoost Regressor   → next 10-min % price change
  - No LSTM (too slow for 10-min retraining)
  - No news/FII/DII (intraday features only)

Features (all lookahead-safe, using only bar T to predict bar T+1):
  - Returns: 1-bar, 2-bar, 3-bar, 5-bar, 10-bar log returns
  - Volume: ratio to rolling mean, z-score
  - Bar shape: body %, upper wick %, lower wick %
  - VWAP deviation
  - RSI (7-period on 10-min closes)
  - EMA crossovers (5 vs 15 bar)
  - Cumulative intraday return (from open)
  - Time features: bars since open, hour, is-morning/afternoon
  - Range expansion: ATR ratio to recent average

Session accuracy is stored in a JSON sidecar file so the live UI
can always read how well the model is doing today.
"""

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from config import MODEL_DIR

logger = logging.getLogger(__name__)

_SESSION_DIR = MODEL_DIR  # store JSON sidecars alongside .joblib files


# ── Feature engineering on 10-min bars ───────────────────────────────────────

def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from a 10-min OHLCV DataFrame.
    All features are computed from data available at bar close — no lookahead.
    """
    if len(df) < 10:
        return pd.DataFrame()

    f = pd.DataFrame(index=df.index)
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # ── Returns ─────────────────────────────────────────────────────────────
    for n in [1, 2, 3, 5, 10]:
        f[f"ret_{n}"] = np.log(c / c.shift(n))

    # ── Bar shape ────────────────────────────────────────────────────────────
    rng = (h - lo).replace(0, np.nan)
    f["body_pct"]  = (c - o).abs() / rng
    f["upper_wick"] = (h - c.clip(upper=h, lower=o)) / rng
    f["lower_wick"] = (c.clip(upper=h, lower=o) - lo) / rng
    f["is_bull_bar"] = (c > o).astype(float)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_ma10 = v.rolling(10, min_periods=3).mean()
    f["vol_ratio"]  = v / vol_ma10.replace(0, np.nan)
    vol_std = v.rolling(10, min_periods=3).std().replace(0, np.nan)
    f["vol_zscore"] = (v - vol_ma10) / vol_std

    # ── VWAP deviation ───────────────────────────────────────────────────────
    typical = (h + lo + c) / 3
    vwap_num = (typical * v).groupby(df.index.date).cumsum()
    vwap_den = v.groupby(df.index.date).cumsum().replace(0, np.nan)
    vwap = vwap_num / vwap_den
    f["vwap_dev"] = (c - vwap) / vwap.replace(0, np.nan)

    # ── RSI (7-period) ───────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(7, min_periods=3).mean()
    loss  = (-delta.clip(upper=0)).rolling(7, min_periods=3).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_7"] = 100 - (100 / (1 + rs))

    # ── EMA crossover ────────────────────────────────────────────────────────
    ema5  = c.ewm(span=5,  adjust=False).mean()
    ema15 = c.ewm(span=15, adjust=False).mean()
    f["ema5_above_ema15"] = (ema5 > ema15).astype(float)
    f["ema5_dev"]  = (c - ema5)  / ema5.replace(0, np.nan)
    f["ema15_dev"] = (c - ema15) / ema15.replace(0, np.nan)

    # ── Cumulative intraday return from open ─────────────────────────────────
    day_open = o.groupby(df.index.date).transform("first")
    f["cum_intraday_ret"] = np.log(c / day_open.replace(0, np.nan))

    # ── ATR ratio ────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr5  = tr.rolling(5, min_periods=2).mean()
    atr20 = tr.rolling(20, min_periods=5).mean()
    f["atr_ratio"] = atr5 / atr20.replace(0, np.nan)  # >1 = expanding range

    # ── Time features ────────────────────────────────────────────────────────
    # Number of 10-min bars since market open (first bar of the day = 0)
    f["bars_since_open"] = df.groupby(df.index.date).cumcount()
    f["hour"] = df.index.hour
    f["is_morning"] = (df.index.hour < 12).astype(float)
    f["is_last_hour"] = (df.index.hour >= 14).astype(float)

    # ── Trend strength ───────────────────────────────────────────────────────
    # Slope of 10-bar linear regression (normalised by price)
    def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
        def _slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0.0
            m = np.polyfit(x, y, 1)[0]
            return m / (y.mean() or 1)
        return series.rolling(window, min_periods=3).apply(_slope, raw=True)

    f["slope_10"] = rolling_slope(c)

    # ── Targets (lookahead — used only for training, not inference) ──────────
    f["target"]        = (c.shift(-1) > c).astype(float)        # direction
    f["target_return"] = np.log(c.shift(-1) / c)                # log-return

    # Last bar has no next bar
    f.iloc[-1, f.columns.get_loc("target")]        = np.nan
    f.iloc[-1, f.columns.get_loc("target_return")] = np.nan

    # ── Clean up ─────────────────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in f.columns if c not in ("target", "target_return")]
    f[feature_cols] = f[feature_cols].ffill().fillna(0.0)

    return f


FEATURE_COLS_INTRADAY = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "body_pct", "upper_wick", "lower_wick", "is_bull_bar",
    "vol_ratio", "vol_zscore",
    "vwap_dev", "rsi_7",
    "ema5_above_ema15", "ema5_dev", "ema15_dev",
    "cum_intraday_ret",
    "atr_ratio",
    "bars_since_open", "hour", "is_morning", "is_last_hour",
    "slope_10",
]

_XGB_PARAMS = {
    "n_estimators":    100,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":3,
    "eval_metric":     "logloss",
    "random_state":    42,
    "n_jobs":          -1,
    "verbosity":       0,
}

_XGB_REG_PARAMS = {**{k: v for k, v in _XGB_PARAMS.items() if k != "eval_metric"},
                   "eval_metric": "mae"}


# ── Model paths ───────────────────────────────────────────────────────────────

def _clf_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_intraday_clf.joblib"

def _reg_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_intraday_reg.joblib"

def _session_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_intraday_session.json"


# ── Training ──────────────────────────────────────────────────────────────────

def train_intraday(ticker: str, bars_df: pd.DataFrame) -> dict[str, Any]:
    """
    Train the intraday classifier + regressor on `bars_df` (10-min OHLCV).
    Saves models to disk and returns metadata.
    Designed to complete in < 500ms for ~190 bars.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    feat_df = build_intraday_features(bars_df)

    if feat_df.empty or len(feat_df) < 20:
        return {"error": "not enough bars to train", "trained_bars": 0}

    avail = [c for c in FEATURE_COLS_INTRADAY if c in feat_df.columns]
    train = feat_df.dropna(subset=["target", "target_return"])

    if len(train) < 15:
        return {"error": "not enough labelled bars", "trained_bars": 0}

    X     = train[avail]
    y_clf = train["target"].astype(int)
    y_reg = train["target_return"].astype(float)

    clf = XGBClassifier(**_XGB_PARAMS)
    clf.fit(X, y_clf)

    reg = XGBRegressor(**_XGB_REG_PARAMS)
    reg.fit(X, y_reg)

    joblib.dump({"model": clf, "feature_cols": avail}, _clf_path(symbol))
    joblib.dump({"model": reg, "feature_cols": avail}, _reg_path(symbol))

    logger.debug("Intraday model trained for %s on %d bars", symbol, len(train))
    return {"trained_bars": len(train), "feature_cols": avail}


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_bar(ticker: str, bars_df: pd.DataFrame) -> dict[str, Any]:
    """
    Predict the next 10-minute bar using the intraday model.

    Returns:
        direction        — 1 = up, 0 = down
        confidence       — P(predicted class)
        predicted_price  — estimated next close (₹)
        predicted_return — estimated % change
        upper / lower    — ±1 ATR confidence bands
        last_price       — current close price
        next_time        — expected time of next bar (str "HH:MM")
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    clf_p  = _clf_path(symbol)
    reg_p  = _reg_path(symbol)

    if not clf_p.exists() or not reg_p.exists():
        return {"error": "intraday model not trained yet"}

    feat_df = build_intraday_features(bars_df)
    if feat_df.empty:
        return {"error": "could not build features"}

    clf_saved = joblib.load(clf_p)
    reg_saved = joblib.load(reg_p)
    clf: XGBClassifier = clf_saved["model"]
    reg: XGBRegressor  = reg_saved["model"]
    cols = clf_saved["feature_cols"]

    avail = [c for c in cols if c in feat_df.columns]
    last_feat = feat_df.iloc[[-1]][avail].fillna(0.0)
    last_price = float(bars_df["close"].iloc[-1])

    # Classifier
    proba = clf.predict_proba(last_feat)[0]
    classes = list(clf.classes_)
    up_idx  = classes.index(1) if 1 in classes else 1
    up_prob = float(proba[up_idx])
    direction  = 1 if up_prob >= 0.5 else 0
    confidence = up_prob if direction == 1 else (1 - up_prob)

    # Regressor
    log_ret    = float(reg.predict(last_feat)[0])
    # Cap to ±2% per 10-min bar (extreme values are noise)
    log_ret    = max(-0.02, min(0.02, log_ret))
    pred_ret   = math.expm1(log_ret) * 100
    pred_price = round(last_price * math.exp(log_ret), 2)

    # ATR for band width
    feat_atr   = float(feat_df["atr_ratio"].iloc[-1]) if "atr_ratio" in feat_df.columns else 1.0
    atr_abs    = float(bars_df[["high", "low"]].diff().abs().mean().mean()) * feat_atr
    band       = atr_abs * (1.2 - confidence * 0.4)  # tighter when more confident

    # Next bar timestamp
    last_ts    = bars_df.index[-1]
    next_ts    = last_ts + pd.Timedelta(minutes=10)
    next_time  = next_ts.strftime("%H:%M") if hasattr(next_ts, "strftime") else "—"

    return {
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "up_probability":  round(up_prob, 4),
        "predicted_price": pred_price,
        "predicted_return":round(pred_ret, 3),
        "upper_band":      round(pred_price + band, 2),
        "lower_band":      round(pred_price - band, 2),
        "last_price":      round(last_price, 2),
        "next_time":       next_time,
    }


# ── Session accuracy tracking ─────────────────────────────────────────────────

def _load_session(ticker: str) -> dict:
    path = _session_path(ticker)
    if path.exists():
        try:
            with open(path) as f:
                s = json.load(f)
            if s.get("date") == date.today().isoformat():
                return s
        except Exception:
            pass
    return {"date": date.today().isoformat(), "predictions": []}


def _save_session(ticker: str, session: dict) -> None:
    with open(_session_path(ticker), "w") as f:
        json.dump(session, f)


def record_prediction(ticker: str, pred: dict) -> None:
    """Store a prediction in today's session log (before we know the outcome)."""
    session = _load_session(ticker)
    session["predictions"].append({
        "time":             datetime.now().strftime("%H:%M:%S"),
        "predicted_dir":    pred.get("direction"),
        "predicted_price":  pred.get("predicted_price"),
        "actual_price":     None,
        "correct":          None,
    })
    _save_session(ticker, session)


def record_actual(ticker: str, actual_price: float) -> bool | None:
    """
    Fill in the actual price for the most recent unresolved prediction.
    Returns True/False if prediction was correct, None if nothing to fill.
    """
    session = _load_session(ticker)
    for p in reversed(session["predictions"]):
        if p["actual_price"] is None:
            prev_price = float(p["predicted_price"] or actual_price)
            actual_dir = 1 if actual_price > prev_price else 0
            p["actual_price"] = actual_price
            p["correct"]      = (actual_dir == p["predicted_dir"])
            _save_session(ticker, session)
            return p["correct"]
    return None


def get_session_accuracy(ticker: str) -> dict:
    """Return today's running accuracy stats."""
    session = _load_session(ticker)
    resolved = [p for p in session["predictions"] if p["correct"] is not None]
    total    = len(resolved)
    correct  = sum(1 for p in resolved if p["correct"])
    return {
        "total":     total,
        "correct":   correct,
        "accuracy":  round(correct / total, 4) if total > 0 else None,
        "predictions": session["predictions"],
    }
