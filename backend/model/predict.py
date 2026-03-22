import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from model.train import load_model, load_regressor
from features.feature_builder import get_feature_columns

logger = logging.getLogger(__name__)


def generate_predictions(
    ticker: str,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate historical predictions for the entire price history.

    Returns a DataFrame with columns:
        date, actual_price, predicted_direction (1/0),
        prediction_confidence, upper_band, lower_band
    """
    model, metadata = load_model(ticker)
    feature_cols = metadata.get("feature_cols", get_feature_columns(feature_df))

    available_cols = [c for c in feature_cols if c in feature_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning("Missing feature columns during prediction: %s", missing_cols)

    predict_df = feature_df.copy()
    predict_df[available_cols] = predict_df[available_cols].ffill().fillna(0.0)
    predict_df = predict_df.dropna(subset=["Close"])

    if predict_df.empty:
        logger.warning("No rows available for prediction after NaN drop")
        return pd.DataFrame()

    X = predict_df[available_cols]

    pred_classes = model.predict(X)
    pred_proba   = model.predict_proba(X)

    classes = list(model.classes_)
    up_idx  = classes.index(1) if 1 in classes else 1
    confidence = pred_proba[:, up_idx]

    prediction_confidence = [
        pred_proba[i, up_idx] if pred_classes[i] == 1 else 1 - pred_proba[i, up_idx]
        for i in range(len(pred_classes))
    ]

    atr          = predict_df.get("atr_14", pd.Series(0.0, index=predict_df.index))
    actual_price = predict_df["Close"]

    result_df = pd.DataFrame(
        {
            "date": predict_df.index,
            "actual_price": actual_price.values,
            "predicted_direction": pred_classes.astype(int),
            "prediction_confidence": prediction_confidence,
            "raw_up_probability": confidence,
            "upper_band": (actual_price + (1 - pd.Series(prediction_confidence, index=predict_df.index)) * atr).values,
            "lower_band": (actual_price - (1 - pd.Series(prediction_confidence, index=predict_df.index)) * atr).values,
        }
    )

    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.sort_values("date").reset_index(drop=True)

    logger.info("Generated %d predictions for %s", len(result_df), ticker)
    return result_df


def predict_tomorrow(
    ticker: str,
    feature_df: pd.DataFrame,
) -> dict:
    """
    Generate a precise prediction for the NEXT trading day using:
        - XGBoost Classifier (direction probability)
        - XGBoost Regressor  (price magnitude)
        - LSTM               (sequential direction probability)

    Ensemble:
        up_prob = 0.55 * xgb_prob + 0.45 * lstm_prob   (if LSTM available)
                = xgb_prob                               (if LSTM not yet trained)

    Returns a dict:
        {
            "predicted_price":      float,   # point estimate
            "predicted_price_low":  float,   # lower confidence band
            "predicted_price_high": float,   # upper confidence band
            "direction":            int,     # 1 = up, 0 = down
            "confidence":           float,   # 0.5 – 1.0
            "xgb_up_prob":          float,
            "lstm_up_prob":         float | None,
            "predicted_return_pct": float,   # expected % move
            "model":                str,
        }
    """
    model, metadata = load_model(ticker)
    feature_cols = metadata.get("feature_cols", get_feature_columns(feature_df))
    available_cols = [c for c in feature_cols if c in feature_df.columns]

    df = feature_df.copy()
    df[available_cols] = df[available_cols].ffill().fillna(0.0)
    df = df.dropna(subset=["Close"])

    if df.empty:
        return {}

    last_row    = df.iloc[[-1]]  # Keep as DataFrame for predict()
    last_price  = float(df["Close"].iloc[-1])
    atr         = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else last_price * 0.015

    X = last_row[available_cols]

    # ── XGBoost classifier probability ───────────────────────────────────────
    proba  = model.predict_proba(X)[0]
    classes = list(model.classes_)
    up_idx  = classes.index(1) if 1 in classes else 1
    xgb_up_prob = float(proba[up_idx])

    # ── LSTM probability ──────────────────────────────────────────────────────
    lstm_up_prob: float | None = None
    try:
        from model.lstm_model import predict_lstm, lstm_exists
        if lstm_exists(ticker):
            lstm_up_prob = predict_lstm(ticker, df, available_cols)
    except Exception as exc:
        logger.debug("LSTM prediction skipped for %s: %s", ticker, exc)

    # ── Ensemble direction probability ────────────────────────────────────────
    if lstm_up_prob is not None:
        ensemble_up_prob = 0.55 * xgb_up_prob + 0.45 * lstm_up_prob
        model_label = "XGBoost+LSTM"
    else:
        ensemble_up_prob = xgb_up_prob
        model_label = "XGBoost"

    direction = 1 if ensemble_up_prob >= 0.5 else 0
    confidence = ensemble_up_prob if direction == 1 else (1 - ensemble_up_prob)

    # ── XGBoost regressor: predicted log-return magnitude ────────────────────
    predicted_return_pct: float
    regressor, reg_cols = load_regressor(ticker)

    if regressor is not None:
        reg_available = [c for c in (reg_cols or available_cols) if c in df.columns]
        X_reg = last_row[reg_available]
        log_ret = float(regressor.predict(X_reg)[0])
        # Convert log-return to price
        raw_return_pct = (np.exp(log_ret) - 1) * 100
        # Cap per-day move at ±3% to prevent extreme single-day predictions
        predicted_return_pct = max(-3.0, min(3.0, raw_return_pct))
    else:
        # Fall back: use median recent daily move × direction
        recent_returns = df["Close"].pct_change().tail(20).dropna()
        median_move    = float(recent_returns.abs().median()) if len(recent_returns) > 5 else 0.006
        predicted_return_pct = median_move * 100 * (1 if direction == 1 else -1)

    # ── Build price prediction ────────────────────────────────────────────────
    predicted_price = last_price * (1 + predicted_return_pct / 100)

    # Uncertainty bands: ATR-based, narrower when confidence is higher
    # At 75% confidence, band = 0.5× ATR on each side
    # At 50% confidence, band = 1.0× ATR on each side
    uncertainty_mult = 1.0 - (confidence - 0.5) * 2 * 0.5  # 1.0 → 0.5 as conf 50%→100%
    band_width       = atr * max(0.3, uncertainty_mult)

    predicted_price_low  = round(predicted_price - band_width, 2)
    predicted_price_high = round(predicted_price + band_width, 2)

    return {
        "predicted_price":      float(round(predicted_price, 2)),
        "predicted_price_low":  float(predicted_price_low),
        "predicted_price_high": float(predicted_price_high),
        "direction":            int(direction),
        "confidence":           float(round(confidence, 4)),
        "xgb_up_prob":          float(round(xgb_up_prob, 4)),
        "lstm_up_prob":         float(round(lstm_up_prob, 4)) if lstm_up_prob is not None else None,
        "predicted_return_pct": float(round(predicted_return_pct, 3)),
        "model":                str(model_label),
    }


def generate_future_predictions(
    ticker: str,
    feature_df: pd.DataFrame,
    days: int = 30,
) -> pd.DataFrame:
    """
    Generate projected price predictions for the next N trading days.

    Day 1 (tomorrow) uses the precise `predict_tomorrow()` ensemble output.
    Days 2–N use the autoregressive walk anchored to the Day 1 prediction.

    Returns a DataFrame with:
        date, projected_price, predicted_direction, confidence,
        upper_band, lower_band, predicted_price (Day 1 specific)
    """
    model, metadata = load_model(ticker)
    feature_cols    = metadata.get("feature_cols", get_feature_columns(feature_df))
    available_cols  = [c for c in feature_cols if c in feature_df.columns]

    df = feature_df.copy()
    df[available_cols] = df[available_cols].ffill().fillna(0.0)
    df = df.dropna(subset=["Close"])

    if df.empty:
        logger.warning("No data available for future prediction")
        return pd.DataFrame()

    last_row   = df.iloc[-1].copy()
    last_price = float(last_row["Close"])
    last_date  = df.index[-1]

    # Stable base move: median daily move capped at 1%
    daily_returns    = df["Close"].pct_change().tail(60).dropna()
    median_move_pct  = float(daily_returns.abs().median()) if len(daily_returns) > 5 else 0.006
    avg_daily_move_pct = min(median_move_pct, 0.010)

    # 20-day MA for mean-reversion anchor
    ma20 = float(df["Close"].tail(20).mean())

    # ── Day 1: use precise ensemble prediction ────────────────────────────────
    tomorrow_pred = predict_tomorrow(ticker, feature_df)

    future_rows: list[dict] = []
    current_price    = last_price
    current_features = last_row.copy()

    trading_day    = last_date
    days_generated = 0

    while days_generated < days:
        trading_day = trading_day + timedelta(days=1)
        if trading_day.weekday() >= 5:
            continue

        days_generated += 1

        if days_generated == 1 and tomorrow_pred:
            # Use the precise ensemble prediction for tomorrow
            pred_class  = tomorrow_pred["direction"]
            confidence  = tomorrow_pred["confidence"]
            next_price  = tomorrow_pred["predicted_price"]
            atr_val     = float(current_features.get("atr_14", current_price * 0.015))
            # Bands from the ensemble prediction
            upper = tomorrow_pred["predicted_price_high"]
            lower = tomorrow_pred["predicted_price_low"]
        else:
            # Autoregressive for days 2+
            X = pd.DataFrame([current_features[available_cols]])
            pred_class = int(model.predict(X)[0])
            pred_proba = model.predict_proba(X)[0]

            classes = list(model.classes_)
            up_idx  = classes.index(1) if 1 in classes else 1
            up_prob = float(pred_proba[up_idx])
            confidence = up_prob if pred_class == 1 else 1 - up_prob

            # Mean-reversion blend
            deviation_from_ma = (current_price - ma20) / ma20
            mean_reversion_pull = -deviation_from_ma * 0.05

            direction_multiplier = 1.0 if pred_class == 1 else -1.0
            raw_move_pct    = avg_daily_move_pct * direction_multiplier
            blended_move_pct = raw_move_pct * 0.7 + mean_reversion_pull * 0.3

            next_price = current_price * (1 + blended_move_pct)

            atr_val = float(current_features.get("atr_14", current_price * 0.015))
            uncertainty_factor = 1 + (days_generated * 0.06)
            upper = next_price + (1 - confidence) * atr_val * uncertainty_factor
            lower = next_price - (1 - confidence) * atr_val * uncertainty_factor

        future_rows.append({
            "date": trading_day,
            "projected_price": round(next_price, 2),
            "predicted_direction": pred_class,
            "prediction_confidence": round(confidence, 4),
            "upper_band": round(upper, 2),
            "lower_band": round(lower, 2),
        })

        # Update rolling features for next iteration
        current_features = current_features.copy()
        current_features["Close"] = next_price
        if days_generated > 1:
            pct_chg = (next_price - current_price) / current_price if current_price else 0
            current_features["daily_return"] = pct_chg
        current_price = next_price

    future_df = pd.DataFrame(future_rows)
    logger.info("Generated %d future predictions for %s", len(future_df), ticker)
    return future_df
