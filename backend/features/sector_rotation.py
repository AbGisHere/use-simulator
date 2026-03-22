"""
Sector rotation signal builder.

Each NSE stock belongs to a sector. If the broader sector is gaining momentum,
individual stocks in that sector are more likely to rise (and vice versa).

This module:
1. Maps each ticker to its Nifty sector index (fetched via yfinance)
2. Computes 5-day and 20-day return for the sector index
3. Returns a DataFrame that can be merged into the main feature matrix

The sector indices used are those traded on NSE/BSE and available via yfinance.
All computations are lag-safe (sector return up to and including day T is used
as a feature for predicting day T+1 price).
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Sector index mapping ──────────────────────────────────────────────────────
# Maps NSE ticker → yfinance sector index symbol
# Using ^CNXIT-style NSE indices where available; fallback to Nifty 50 (^NSEI)

SECTOR_INDEX_MAP: dict[str, str] = {
    # IT
    "TCS": "^CNXIT", "INFY": "^CNXIT", "WIPRO": "^CNXIT",
    "HCLTECH": "^CNXIT", "TECHM": "^CNXIT", "MPHASIS": "^CNXIT",

    # Banking
    "HDFCBANK": "^NSEBANK", "ICICIBANK": "^NSEBANK", "SBIN": "^NSEBANK",
    "AXISBANK": "^NSEBANK", "KOTAKBANK": "^NSEBANK", "BAJFINANCE": "^NSEBANK",
    "BAJAJFINSV": "^NSEBANK", "SBICARD": "^NSEBANK",
    "HDFCLIFE": "^NSEBANK", "SBILIFE": "^NSEBANK",

    # FMCG
    "HINDUNILVR": "^CNXFMCG", "NESTLEIND": "^CNXFMCG", "DABUR": "^CNXFMCG",
    "MARICO": "^CNXFMCG", "GODREJCP": "^CNXFMCG", "ITC": "^CNXFMCG",

    # Pharma
    "SUNPHARMA": "^CNXPHARMA", "DRREDDY": "^CNXPHARMA",
    "CIPLA": "^CNXPHARMA", "DIVISLAB": "^CNXPHARMA",
    "APOLLOHOSP": "^CNXPHARMA",

    # Auto
    "MARUTI": "^CNXAUTO", "TATAMOTORS": "^CNXAUTO", "M&M": "^CNXAUTO",
    "HEROMOTOCO": "^CNXAUTO", "BAJAJ-AUTO": "^CNXAUTO", "EICHERMOT": "^CNXAUTO",

    # Metal
    "TATASTEEL": "^CNXMETAL", "JSWSTEEL": "^CNXMETAL",
    "HINDALCO": "^CNXMETAL", "VEDL": "^CNXMETAL", "COALINDIA": "^CNXMETAL",

    # Energy / Oil
    "RELIANCE": "^CNXENERGY", "ONGC": "^CNXENERGY", "BPCL": "^CNXENERGY",
    "IOC": "^CNXENERGY", "HINDPETRO": "^CNXENERGY",
    "NTPC": "^CNXENERGY", "POWERGRID": "^CNXENERGY", "TATAPOWER": "^CNXENERGY",
    "ADANIGREEN": "^CNXENERGY",

    # Infra / Capital Goods
    "LT": "^CNXINFRA", "ADANIPORTS": "^CNXINFRA", "ADANIENT": "^CNXINFRA",

    # Realty
    "DLF": "^CNXREALTY",

    # Media / Entertainment
    "PVR": "^CNXMEDIA", "PVRINOX": "^CNXMEDIA",
    "INOXLEISUR": "^CNXMEDIA", "ZEEL": "^CNXMEDIA",

    # Aviation / Hotels (use Nifty 50 as proxy — no dedicated index)
    "INDIGO": "^NSEI", "SPICEJET": "^NSEI",
    "INDHOTEL": "^NSEI", "LEMONTREE": "^NSEI",

    # Chemicals
    "PIDILITIND": "^NSEI", "AARTIIND": "^NSEI", "SRF": "^NSEI", "RAIN": "^NSEI",

    # New-age Tech / Consumer
    "ZOMATO": "^NSEI", "NYKAA": "^NSEI", "TITAN": "^NSEI",

    # Defence
    "HAL": "^NSEI", "BEL": "^NSEI",

    # Agri
    "COROMANDEL": "^NSEI", "CHAMBLFERT": "^NSEI",

    # Cement
    "ULTRACEMCO": "^NSEI", "AMBUJACEM": "^NSEI", "ACC": "^NSEI",

    # Telecom
    "BHARTIARTL": "^NSEI",
}

_DEFAULT_INDEX = "^NSEI"


def _get_sector_index(ticker: str) -> str:
    return SECTOR_INDEX_MAP.get(ticker.upper().replace(".NS", "").replace(".BO", ""), _DEFAULT_INDEX)


def fetch_sector_rotation(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """
    Fetch sector index data and compute rotation signals for `ticker`.

    Returns a DataFrame indexed by date with columns:
        sector_return_5d   — 5-day return of the sector index (%)
        sector_return_20d  — 20-day return of the sector index (%)
        sector_above_ma50  — 1 if sector index > its 50-day MA (trend filter)
        sector_momentum    — sector_return_5d rank vs last 252 days (0–1 percentile)

    All signals use data available as of market close on day T, so no lookahead.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    index_sym = _get_sector_index(symbol)

    logger.info("Fetching sector index %s for %s…", index_sym, symbol)

    try:
        if start and end:
            idx = yf.download(index_sym, start=start, end=end, progress=False, auto_adjust=True)
        else:
            idx = yf.download(index_sym, period=period, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.warning("Sector index download failed for %s: %s", index_sym, exc)
        return pd.DataFrame(columns=["sector_return_5d", "sector_return_20d",
                                      "sector_above_ma50", "sector_momentum"])

    if idx.empty:
        logger.warning("Empty sector index data for %s", index_sym)
        return pd.DataFrame(columns=["sector_return_5d", "sector_return_20d",
                                      "sector_above_ma50", "sector_momentum"])

    # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = [col[0] for col in idx.columns]

    close = idx["Close"] if "Close" in idx.columns else idx.iloc[:, 0]
    close = close.ffill()

    df = pd.DataFrame(index=close.index)
    df["sector_return_5d"]  = close.pct_change(5) * 100
    df["sector_return_20d"] = close.pct_change(20) * 100

    ma50 = close.rolling(50, min_periods=25).mean()
    df["sector_above_ma50"] = (close > ma50).astype(float)

    # Rolling percentile rank of 5-day return over last 252 days
    def rolling_rank(series: pd.Series, window: int = 252) -> pd.Series:
        def _rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        return series.rolling(window, min_periods=30).apply(_rank, raw=False)

    df["sector_momentum"] = rolling_rank(df["sector_return_5d"])

    df = df.ffill().fillna(0.0)
    df.index = pd.to_datetime(df.index).normalize()

    logger.info("Sector rotation: %d rows for %s (index %s)", len(df), symbol, index_sym)
    return df
