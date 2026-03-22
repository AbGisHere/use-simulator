import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from model.train import load_model
from features.feature_builder import get_feature_columns

logger = logging.getLogger(__name__)


def generate_predictions(
    ticker: str,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for the entire historical period.

    Returns a DataFrame with columns:
        date, actual_price, predicted_direction (1/0),
        prediction_confidence, upper_band, lower_band
    """
    model, metadata = load_model(ticker)
    feature_cols = metadata.get("feature_cols", get_feature_columns(feature_df))

    # Only use columns that were present during training
    available_cols = [c for c in feature_cols if c in feature_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning("Missing feature columns during prediction: %s", missing_cols)

    # Apply same NaN handling as training — ffill then zero-fill so no rows are lost
    predict_df = feature_df.copy()
    predict_df[available_cols] = predict_df[available_cols].ffill().fillna(0.0)

    # Only drop rows missing the actual price (can't generate a meaningful prediction)
    predict_df = predict_df.dropna(subset=["Close"])

    if predict_df.empty:
        logger.warning("No rows available for prediction after NaN drop")
        return pd.DataFrame()

    X = predict_df[available_cols]

    # Predict class and probability
    pred_classes = model.predict(X)
    pred_proba = model.predict_proba(X)

    # XGBoost classes are [0, 1]; proba[:, 1] = P(up)
    classes = list(model.classes_)
    up_idx = classes.index(1) if 1 in classes else 1
    confidence = pred_proba[:, up_idx]

    # Confidence for each prediction: P(predicted class)
    prediction_confidence = [
        pred_proba[i, up_idx] if pred_classes[i] == 1 else 1 - pred_proba[i, up_idx]
        for i in range(len(pred_classes))
    ]

    # ATR-based confidence bands
    atr = predict_df.get("atr_14", pd.Series(0.0, index=predict_df.index))
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


def generate_future_predictions(
    ticker: str,
    feature_df: pd.DataFrame,
    days: int = 30,
) -> pd.DataFrame:
    """
    Generate projected price predictions for the next N trading days.

    Strategy:
        - Start from the last known actual price
        - Each day: predict direction + confidence using the model
        - Project price using predicted direction * recent average daily move
        - Build rolling features from the projected prices so each day
          feeds into the next (autoregressive forecasting)

    Returns a DataFrame with:
        date, projected_price, predicted_direction, confidence, upper_band, lower_band
    """
    model, metadata = load_model(ticker)
    feature_cols = metadata.get("feature_cols", get_feature_columns(feature_df))
    available_cols = [c for c in feature_cols if c in feature_df.columns]

    # Work from the last known row of actual data
    df = feature_df.copy()
    df[available_cols] = df[available_cols].ffill().fillna(0.0)
    df = df.dropna(subset=["Close"])

    if df.empty:
        logger.warning("No data available for future prediction")
        return pd.DataFrame()

    last_row = df.iloc[-1].copy()
    last_price = float(last_row["Close"])
    last_date = df.index[-1]

    # Use median absolute daily return (not std) as the base move size.
    # std over-estimates because it includes outliers; median is more stable.
    daily_returns = df["Close"].pct_change().tail(60).dropna()
    median_move_pct = float(daily_returns.abs().median()) if len(daily_returns) > 5 else 0.006
    # Cap at 1% per day — prevents extreme compounding over 30 days
    avg_daily_move_pct = min(median_move_pct, 0.010)

    # 20-day moving average for mean-reversion anchor
    ma20 = float(df["Close"].tail(20).mean())

    future_rows: list[dict] = []
    current_price = last_price
    current_features = last_row.copy()

    # Generate next N trading days (skip weekends)
    trading_day = last_date
    days_generated = 0

    while days_generated < days:
        trading_day = trading_day + timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if trading_day.weekday() >= 5:
            continue

        # Predict using current feature vector
        X = pd.DataFrame([current_features[available_cols]])
        pred_class = int(model.predict(X)[0])
        pred_proba = model.predict_proba(X)[0]

        classes = list(model.classes_)
        up_idx = classes.index(1) if 1 in classes else 1
        up_prob = float(pred_proba[up_idx])
        confidence = up_prob if pred_class == 1 else 1 - up_prob

        # Blend model direction with gentle mean-reversion.
        # If price has drifted >5% from 20-day MA, pull it back slightly.
        # This prevents unrealistic runaway trends in the forecast.
        deviation_from_ma = (current_price - ma20) / ma20
        mean_reversion_pull = -deviation_from_ma * 0.05  # 5% of deviation per day

        direction_multiplier = 1.0 if pred_class == 1 else -1.0
        raw_move_pct = avg_daily_move_pct * direction_multiplier
        blended_move_pct = raw_move_pct * 0.7 + mean_reversion_pull * 0.3

        price_change = current_price * blended_move_pct
        next_price = current_price + price_change

        # ATR-based uncertainty bands (widen over time as uncertainty grows)
        atr = float(current_features.get("atr_14", current_price * 0.015))
        uncertainty_factor = 1 + (days_generated * 0.06)  # bands widen each day
        upper = next_price + (1 - confidence) * atr * uncertainty_factor
        lower = next_price - (1 - confidence) * atr * uncertainty_factor

        future_rows.append({
            "date": trading_day,
            "projected_price": round(next_price, 2),
            "predicted_direction": pred_class,
            "prediction_confidence": round(confidence, 4),
            "upper_band": round(upper, 2),
            "lower_band": round(lower, 2),
        })

        # Update rolling features for next iteration (simplified update)
        current_features = current_features.copy()
        current_features["Close"] = next_price
        current_features["daily_return"] = blended_move_pct
        current_price = next_price
        days_generated += 1

    future_df = pd.DataFrame(future_rows)
    logger.info("Generated %d future predictions for %s", len(future_df), ticker)
    return future_df
