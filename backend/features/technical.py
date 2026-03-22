import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on an OHLCV DataFrame.

    Input columns: Open, High, Low, Close, Volume
    Returns the same DataFrame with additional indicator columns.
    No lookahead bias: all indicators are computed on data available at EOD.
    """
    if df.empty or len(df) < 30:
        logger.warning("DataFrame too short (%d rows) for full indicator computation", len(df))
        return df

    result = df.copy()

    # ── RSI (14-period) ──────────────────────────────────────────────────────
    rsi = ta.rsi(result["Close"], length=14)
    result["rsi_14"] = rsi

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    macd = ta.macd(result["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        result["macd"] = macd.get("MACD_12_26_9")
        result["macd_signal"] = macd.get("MACDs_12_26_9")
        result["macd_hist"] = macd.get("MACDh_12_26_9")

    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────────
    bbands = ta.bbands(result["Close"], length=20, std=2)
    if bbands is not None:
        result["bb_upper"] = bbands.get("BBU_20_2.0")
        result["bb_middle"] = bbands.get("BBM_20_2.0")
        result["bb_lower"] = bbands.get("BBL_20_2.0")
        bb_middle_safe = result["bb_middle"].astype(float)
        bb_middle_safe = bb_middle_safe.where(bb_middle_safe != 0, np.nan)
        bb_range = (result["bb_upper"].astype(float) - result["bb_lower"].astype(float))
        bb_range_safe = bb_range.where(bb_range != 0, np.nan)
        result["bb_upper"] = result["bb_upper"].astype(float)
        result["bb_middle"] = bb_middle_safe
        result["bb_lower"] = result["bb_lower"].astype(float)
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / bb_middle_safe
        result["bb_pct"] = (result["Close"].astype(float) - result["bb_lower"]) / bb_range_safe

    # ── EMAs ─────────────────────────────────────────────────────────────────
    for period in [9, 21, 50, 200]:
        result[f"ema_{period}"] = ta.ema(result["Close"], length=period)

    # EMA crossover signals
    result["ema_9_21_cross"] = (result["ema_9"] > result["ema_21"]).astype(int)
    result["price_above_ema50"] = (result["Close"] > result["ema_50"]).astype(int)
    result["price_above_ema200"] = (result["Close"] > result["ema_200"]).astype(int)

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    atr = ta.atr(result["High"], result["Low"], result["Close"], length=14)
    result["atr_14"] = atr

    # Normalised ATR (ATR as % of close)
    result["atr_pct"] = result["atr_14"] / result["Close"].replace(0, float("nan"))

    # ── OBV (On Balance Volume) ───────────────────────────────────────────────
    obv = ta.obv(result["Close"], result["Volume"])
    result["obv"] = obv

    # OBV momentum (rate of change over 10 days)
    result["obv_roc_10"] = result["obv"].pct_change(10)

    # ── Returns ───────────────────────────────────────────────────────────────
    result["daily_return"] = result["Close"].pct_change(1)
    result["return_5d"] = result["Close"].pct_change(5)
    result["return_10d"] = result["Close"].pct_change(10)
    result["return_30d"] = result["Close"].pct_change(30)

    # ── Volatility ────────────────────────────────────────────────────────────
    result["volatility_30d"] = result["daily_return"].rolling(30).std()

    # ── Volume indicators ─────────────────────────────────────────────────────
    result["volume_sma_20"] = result["Volume"].rolling(20).mean()
    result["volume_ratio"] = result["Volume"] / result["volume_sma_20"].replace(0, float("nan"))

    # ── Price momentum ────────────────────────────────────────────────────────
    result["high_low_pct"] = (result["High"] - result["Low"]) / result["Close"].replace(0, float("nan"))
    result["close_open_pct"] = (result["Close"] - result["Open"]) / result["Open"].replace(0, float("nan"))

    # ── Stochastic RSI ────────────────────────────────────────────────────────
    stochrsi = ta.stochrsi(result["Close"])
    if stochrsi is not None:
        result["stochrsi_k"] = stochrsi.get("STOCHRSIk_14_14_3_3")
        result["stochrsi_d"] = stochrsi.get("STOCHRSId_14_14_3_3")

    logger.info("Computed %d technical indicators", len(result.columns) - len(df.columns))
    return result
