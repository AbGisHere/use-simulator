"""
Live price fetcher for NSE stocks during trading hours.

Data source: Yahoo Finance via yfinance.
Refresh rate: Yahoo updates quotes every ~15 seconds during market hours.

NSE trading hours (IST):
  Pre-open  : 09:00 – 09:15
  Normal    : 09:15 – 15:30
  Post-close: 15:30 – 16:00  (prices stop changing after 15:30)
  Closed    : All other times + weekends + NSE holidays
"""

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

import yfinance as yf

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# NSE regular trading window
_MARKET_OPEN  = time(9, 15)
_MARKET_CLOSE = time(15, 30)
_PRE_OPEN     = time(9, 0)


def is_trading_hours() -> bool:
    """True if NSE market is currently in normal trading session."""
    now = datetime.now(IST)
    if now.weekday() >= 5:        # Saturday / Sunday
        return False
    t = now.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def is_market_day() -> bool:
    """True if today is a weekday (holiday check not included)."""
    return datetime.now(IST).weekday() < 5


def market_status() -> str:
    """Human-readable market status."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "closed_weekend"
    t = now.time()
    if t < _PRE_OPEN:
        return "pre_pre_open"
    if t < _MARKET_OPEN:
        return "pre_open"
    if t <= _MARKET_CLOSE:
        return "open"
    if t <= time(16, 0):
        return "post_close"
    return "closed"


def fetch_live_price(ticker: str) -> dict:
    """
    Fetch the current live price and intraday stats for an NSE stock.

    Returns a dict with:
        price           — latest trade price (₹)
        prev_close      — previous session's closing price (₹)
        open            — today's open price (₹)
        day_high        — intraday high (₹)
        day_low         — intraday low (₹)
        change          — price - prev_close (₹)
        change_pct      — % change from prev_close
        change_from_open— % change from open (intraday drift)
        volume          — shares traded today
        market_status   — "open" | "pre_open" | "post_close" | "closed_weekend" etc.
        is_trading      — True if market is in normal session
        timestamp       — ISO timestamp of the fetch (IST)
        ticker          — clean symbol (no .NS)
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    yf_symbol = symbol + ".NS"

    try:
        t   = yf.Ticker(yf_symbol)
        fi  = t.fast_info

        price      = _safe_float(getattr(fi, "last_price",      None))
        prev_close = _safe_float(getattr(fi, "previous_close",  None))
        open_price = _safe_float(getattr(fi, "open",            None))
        day_high   = _safe_float(getattr(fi, "day_high",        None))
        day_low    = _safe_float(getattr(fi, "day_low",         None))
        volume     = _safe_int(getattr(fi, "three_month_average_volume", None))

        # Fall back to regular_market_price if last_price is None
        if price is None:
            price = _safe_float(getattr(fi, "regular_market_price", None))

        # Fall back: fetch last 1-day 1-minute bar
        if price is None:
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price      = float(hist["Close"].iloc[-1])
                open_price = open_price or float(hist["Open"].iloc[0])
                day_high   = day_high   or float(hist["High"].max())
                day_low    = day_low    or float(hist["Low"].min())

        if price is None:
            return _error_response(symbol, "Price unavailable from Yahoo Finance")

        if prev_close is None:
            # Get prev_close from 2d history
            hist2 = t.history(period="2d", interval="1d")
            if len(hist2) >= 2:
                prev_close = float(hist2["Close"].iloc[-2])

        change     = round(price - prev_close, 2)           if prev_close else 0.0
        change_pct = round((change / prev_close) * 100, 3)  if prev_close else 0.0
        open_drift = round(((price - open_price) / open_price) * 100, 3) if open_price else 0.0

        status = market_status()

        logger.debug("Live price for %s: ₹%.2f (%+.2f%%)", symbol, price, change_pct)

        return {
            "ticker":            symbol,
            "price":             round(price, 2),
            "prev_close":        round(prev_close, 2) if prev_close else None,
            "open":              round(open_price, 2) if open_price else None,
            "day_high":          round(day_high, 2)   if day_high  else None,
            "day_low":           round(day_low, 2)    if day_low   else None,
            "change":            change,
            "change_pct":        change_pct,
            "change_from_open":  open_drift,
            "volume":            volume,
            "market_status":     status,
            "is_trading":        status == "open",
            "timestamp":         datetime.now(IST).isoformat(),
            "error":             None,
        }

    except Exception as exc:
        logger.warning("Live price fetch failed for %s: %s", symbol, exc)
        return _error_response(symbol, str(exc))


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None   # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _error_response(symbol: str, msg: str) -> dict:
    return {
        "ticker":           symbol,
        "price":            None,
        "prev_close":       None,
        "open":             None,
        "day_high":         None,
        "day_low":          None,
        "change":           None,
        "change_pct":       None,
        "change_from_open": None,
        "volume":           None,
        "market_status":    market_status(),
        "is_trading":       False,
        "timestamp":        datetime.now(IST).isoformat(),
        "error":            msg,
    }
