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
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame for model training and prediction.

    Steps:
        1. Compute technical indicators on price data
        2. Add calendar flags
        3. Compute and merge daily sentiment
        4. Merge sector rotation signals (yfinance sector indices)
        5. Merge FII/DII institutional flow features
        6. Merge delivery % features (NSE bhav copy)
        7. Build prediction targets: direction (classifier) + log return (regressor)
        8. Final cleanup: no lookahead, forward-fill, drop NaNs

    Returns a DataFrame with:
        - All technical features
        - Calendar flags
        - sentiment_score
        - sector rotation features
        - fii/dii features
        - delivery % features
        - target          (1=up next day, 0=down)  — for classifier
        - target_return   (log return next day)     — for regressor
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
        df_dates = df.index.normalize()
        sentiment_series.index = pd.to_datetime(
            [pd.Timestamp(d).replace(tzinfo=None) for d in sentiment_series.index]
        )
        if df_dates.tz is not None:
            df_norm = df_dates.tz_convert(None)
        else:
            df_norm = df_dates
        sentiment_aligned = sentiment_series.reindex(df_norm).ffill().fillna(0.0)
        df["sentiment_score"] = sentiment_aligned.values
    else:
        df["sentiment_score"] = 0.0

    # ── Step 4: Sector rotation signals ──────────────────────────────────────
    if ticker:
        try:
            from features.sector_rotation import fetch_sector_rotation
            sector_df = fetch_sector_rotation(ticker)
            if not sector_df.empty:
                df = _merge_on_date(df, sector_df)
                logger.info("Merged sector rotation features for %s", ticker)
        except Exception as exc:
            logger.warning("Sector rotation fetch failed for %s: %s", ticker, exc)

    # Zero-fill sector columns if they weren't added
    for col in ["sector_return_5d", "sector_return_20d", "sector_above_ma50", "sector_momentum"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Step 5: FII/DII institutional flow ───────────────────────────────────
    try:
        from data.fii_dii_fetcher import fetch_fii_dii
        fii_df = fetch_fii_dii(lookback_days=365)
        if not fii_df.empty:
            df = _merge_on_date(df, fii_df)
            logger.info("Merged FII/DII features (%d rows)", len(fii_df))
    except Exception as exc:
        logger.warning("FII/DII fetch failed: %s", exc)

    for col in ["fii_net", "dii_net", "fii_dii_net", "fii_net_5d", "dii_net_5d", "fii_dii_net_5d"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Step 6: Delivery percentage ───────────────────────────────────────────
    if ticker:
        try:
            from data.delivery_fetcher import fetch_delivery_pct
            deliv_df = fetch_delivery_pct(ticker, lookback_days=365)
            if not deliv_df.empty:
                df = _merge_on_date(df, deliv_df)
                logger.info("Merged delivery_pct features for %s", ticker)
        except Exception as exc:
            logger.warning("Delivery pct fetch failed for %s: %s", ticker, exc)

    for col in ["delivery_pct", "delivery_pct_5d", "delivery_spike"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Step 7: Prediction targets ────────────────────────────────────────────
    next_close = df["Close"].shift(-1)

    # Direction target (classifier): 1 = up, 0 = down
    df["target"] = (next_close > df["Close"]).astype(float)

    # Return target (regressor): log(next_close / close)
    # Using log-return is better than pct_change because it's symmetric and unbounded
    import numpy as np
    df["target_return"] = np.log(next_close / df["Close"])

    # Last row has no next-day target
    df.loc[df.index[-1], "target"] = float("nan")
    df.loc[df.index[-1], "target_return"] = float("nan")

    # ── Step 8: No-lookahead verification & cleanup ───────────────────────────
    feature_cols = [
        c for c in df.columns
        if c not in ("target", "target_return", "Open", "High", "Low", "Close", "Volume")
    ]
    df[feature_cols] = df[feature_cols].ffill()

    core_features = ["rsi_14", "macd", "ema_50", "atr_14"]
    core_available = [c for c in core_features if c in df.columns]
    if core_available:
        df = df.dropna(subset=core_available)

    # Force all feature columns to numeric dtype (pandas 3.x returns object occasionally)
    for col in df.columns:
        if col not in ("target", "target_return") and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Built feature matrix: %d rows × %d columns", len(df), len(df.columns))
    return df


def _merge_on_date(df: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join `right` onto `df` matching on normalised date (tz-stripped).
    Preserves the original df index.
    """
    # Normalise df index to tz-naive date
    df_idx = df.index.normalize()
    if df_idx.tz is not None:
        df_idx = df_idx.tz_convert(None)

    # Normalise right index
    right_idx = pd.to_datetime(right.index).normalize()
    if right_idx.tz is not None:
        right_idx = right_idx.tz_convert(None)
    right_clean = right.copy()
    right_clean.index = right_idx

    # Reindex right onto df's dates
    right_aligned = right_clean.reindex(df_idx)
    right_aligned.index = df.index  # restore original index

    for col in right_aligned.columns:
        df[col] = right_aligned[col].ffill().fillna(0.0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excluding price OHLCV and targets).
    These are the columns fed into the ML model.
    """
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "target", "target_return",
        "Dividends", "Stock Splits",
    }
    return [c for c in df.columns if c not in exclude]
