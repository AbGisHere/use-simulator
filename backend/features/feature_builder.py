import logging
from typing import Any

import pandas as pd

from features.calendar_flags import add_calendar_flags
from features.sentiment import build_daily_sentiment
from features.technical import compute_technical_indicators

logger = logging.getLogger(__name__)


def build_features(
    price_df: pd.DataFrame,
    news_items: list[dict[str, Any]] | None = None,
    nse_announcements: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame for model training and prediction.

    Steps:
        1. Compute technical indicators on price data
        2. Add calendar flags
        3. Compute and merge daily sentiment
        4. Build the prediction target (next day return direction)
        5. Final cleanup: no lookahead, forward-fill, drop NaNs

    Returns a DataFrame with:
        - All technical features
        - Calendar flags
        - sentiment_score
        - target (1=up next day, 0=down), available for training rows only
    """
    if price_df.empty:
        raise ValueError("Price DataFrame is empty — cannot build features")

    # ── Step 1: Technical indicators ─────────────────────────────────────────
    df = compute_technical_indicators(price_df)

    # ── Step 2: Calendar flags ────────────────────────────────────────────────
    df = add_calendar_flags(df)

    # ── Step 3: Sentiment ─────────────────────────────────────────────────────
    all_news: list[dict[str, Any]] = []
    if news_items:
        all_news.extend(news_items)

    # Tag NSE announcements with higher source weight
    if nse_announcements:
        for ann in nse_announcements:
            all_news.append(
                {
                    "headline": ann.get("description", ""),
                    "body": "",
                    "source": "nse announcement",
                    "published_at": ann.get("date"),
                    "score": 0,
                }
            )

    if all_news:
        sentiment_series = build_daily_sentiment(all_news, price_df)
        # Align on normalised dates — strip tz from both sides for reindex matching
        df_dates = df.index.normalize()
        # Strip timezone from sentiment index (already tz-naive from build_daily_sentiment)
        sentiment_series.index = pd.to_datetime(
            [pd.Timestamp(d).replace(tzinfo=None) for d in sentiment_series.index]
        )
        # Strip timezone from price index for alignment
        if df_dates.tz is not None:
            df_norm = df_dates.tz_convert(None)
        else:
            df_norm = df_dates
        sentiment_aligned = sentiment_series.reindex(df_norm).ffill().fillna(0.0)
        df["sentiment_score"] = sentiment_aligned.values
    else:
        df["sentiment_score"] = 0.0

    # ── Step 4: Prediction target ─────────────────────────────────────────────
    # Target: next day close > today close → 1 (up), else 0 (down)
    # Using .shift(-1) is safe because we only use the target for TRAINING
    # During prediction/inference, this column will be NaN for the last row
    next_close = df["Close"].shift(-1)
    df["target"] = (next_close > df["Close"]).astype(float)
    # The last row has no next-day target — mark as NaN
    df.loc[df.index[-1], "target"] = float("nan")

    # ── Step 5: No-lookahead verification & cleanup ───────────────────────────
    # All features must only reference data up to and including the current day.
    # shift(-1) is ONLY used for the target, which is excluded from feature set.

    # Drop rows with too many missing features (first 200 days for indicator warmup)
    feature_cols = [c for c in df.columns if c not in ("target", "Open", "High", "Low", "Close", "Volume")]
    df[feature_cols] = df[feature_cols].ffill()

    # Drop rows where core features are still NaN (indicator warmup period)
    core_features = ["rsi_14", "macd", "ema_50", "atr_14"]
    core_available = [c for c in core_features if c in df.columns]
    if core_available:
        df = df.dropna(subset=core_available)

    # ── Step 6: Force all feature columns to numeric dtype ────────────────────
    # Newer pandas (3.x) can return object dtype from some operations.
    # XGBoost requires int, float, bool or category — never object.
    for col in df.columns:
        if col not in ("target",) and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Built feature matrix: %d rows × %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excluding price OHLCV and target).
    These are the columns fed into the ML model.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "target", "Dividends", "Stock Splits"}
    return [c for c in df.columns if c not in exclude]
