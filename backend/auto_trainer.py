#!/usr/bin/env python3
"""
auto_trainer.py — NSE Simulator Continuous Trainer
===================================================
Two operating modes that switch automatically:

  MARKET OPEN (9:15–15:30 IST, Mon–Fri):
    10-minute intraday training loop
    ┌─ Initial: fetch 5-day bars → train intraday model → predict next bar
    └─ Every 10 min:
         wait for bar to close → record actual price → compute accuracy
         retrain on updated bars → predict next bar → show live table

  MARKET CLOSED:
    Offline replay + overnight Optuna retrain
    ┌─ Replay last 5 days sequentially (train → predict → check → advance)
    ├─ Trigger full Optuna retrain for all tracked stocks
    ├─ Possibly add a random new stock from the NSE universe
    └─ Sleep until 9:10 IST on the next trading day

Cross-platform: works on macOS, Linux, and Windows (PowerShell / cmd).

Usage:
    # macOS / Linux:
    cd backend && source .venv/bin/activate
    python auto_trainer.py

    # Windows (PowerShell):
    cd backend; .venv\\Scripts\\Activate.ps1
    python auto_trainer.py
"""

import os
import platform
import random
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

# ── Platform / timezone setup ──────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak these without touching anything else
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL             = "http://localhost:8000"

# Intraday loop timing
INTRADAY_BAR_SECONDS    = 600   # 10-minute bars
INTRADAY_BAR_BUFFER     = 60    # extra seconds to wait after bar close
INTRADAY_MIN_BARS       = 3     # wait until at least this many bars exist

# Overnight retrain
SLEEP_BETWEEN_STOCKS    = 30    # gap between pipeline runs (seconds)
ADD_NEW_EVERY_N_CYCLES  = 2     # add a random stock every N overnight cycles
MAX_STOCKS              = 20    # cap on total tracked stocks
POLL_TIMEOUT            = 600   # max wait for a pipeline (seconds)
POLL_INTERVAL           = 15    # seconds between pipeline status polls

# ─────────────────────────────────────────────────────────────────────────────
# NSE universe to explore
# ─────────────────────────────────────────────────────────────────────────────

NSE_UNIVERSE = [
    # IT
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MPHASIS",
    # Banks & Finance
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
    "BAJFINANCE", "BAJAJFINSV", "SBICARD", "HDFCLIFE", "SBILIFE",
    # Oil & Energy
    "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO",
    "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN",
    # Infra & Ports
    "ADANIPORTS", "ADANIENT",
    # Auto
    "MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT",
    # Metals
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA",
    # FMCG
    "HINDUNILVR", "NESTLEIND", "DABUR", "MARICO", "GODREJCP", "ITC",
    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    # Telecom
    "BHARTIARTL",
    # Cement
    "ULTRACEMCO", "AMBUJACEM", "ACC",
    # Capital Goods
    "LT",
    # Real Estate
    "DLF",
    # Entertainment / Leisure
    "PVR", "PVRINOX", "INOXLEISUR", "ZEEL", "INDIGO", "SPICEJET",
    "INDHOTEL", "LEMONTREE",
    # Chemicals
    "PIDILITIND", "AARTIIND", "SRF", "RAIN",
    # New-age Tech
    "ZOMATO", "NYKAA",
    # Consumer
    "TITAN",
    # Defence
    "HAL", "BEL",
    # Agri
    "COROMANDEL", "CHAMBLFERT",
]


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers (cross-platform ANSI colour)
# ─────────────────────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    if IS_WINDOWS:
        return bool(
            os.environ.get("WT_SESSION") or
            os.environ.get("TERM_PROGRAM") or
            os.environ.get("COLORTERM")
        )
    return True


_USE_COLOR = _supports_color()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _blue(t):   return _c(t, "34")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")
def _dim(t):    return _c(t, "2")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    icons  = {"INFO": "·", "OK": "v", "WARN": "!", "ERROR": "x", "LIVE": "o", "INTRA": ">"}
    icon   = icons.get(level, "·")
    color  = {"OK": _green, "WARN": _yellow, "ERROR": _red,
               "LIVE": _blue, "INTRA": _cyan}.get(level, lambda x: x)
    print(f"[{_dim(_ts())}] {color(icon + '  ' + msg)}", flush=True)


def separator(title: str = "") -> None:
    width = 70
    line  = "-" * width
    if title:
        pad  = (width - len(title) - 2) // 2
        line = "-" * pad + f" {title} " + "-" * pad
    print(f"\n{_dim(line)}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(path: str, **kw):
    return requests.get(f"{BACKEND_URL}{path}", timeout=kw.pop("timeout", 12), **kw)


def _post(path: str, **kw):
    return requests.post(f"{BACKEND_URL}{path}", timeout=kw.pop("timeout", 60), **kw)


def get_stocks() -> list[dict]:
    try:
        r = _get("/api/stocks")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Could not reach backend: {e}", "ERROR")
        return []


def get_last_updated(ticker: str) -> str | None:
    for s in get_stocks():
        if s["ticker"] == ticker:
            return s.get("last_updated")
    return None


def refresh_stock(ticker: str) -> bool:
    try:
        r = _post(f"/api/stocks/{ticker}/refresh")
        return r.status_code in (200, 202)
    except Exception as e:
        log(f"Refresh failed for {ticker}: {e}", "ERROR")
        return False


def add_stock(ticker: str) -> bool:
    try:
        r = _post("/api/stocks", json={"ticker": ticker})
        return r.status_code in (200, 201, 202)
    except Exception as e:
        log(f"Add failed for {ticker}: {e}", "ERROR")
        return False


def get_live_price(ticker: str) -> dict:
    try:
        r = _get(f"/api/stocks/{ticker}/live", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def wait_for_pipeline(ticker: str, prev_updated: str | None) -> bool:
    deadline = time.time() + POLL_TIMEOUT
    dots = 0
    while time.time() < deadline:
        current = get_last_updated(ticker)
        if current and current != prev_updated:
            print()
            return True
        dots = (dots + 1) % 4
        print(
            f"\r[{_ts()}] {_dim('Waiting for ' + ticker + ' pipeline' + '.' * dots + '   ')}",
            end="", flush=True,
        )
        time.sleep(POLL_INTERVAL)
    print()
    log(f"Timed out waiting for {ticker} ({POLL_TIMEOUT}s)", "WARN")
    return False


def pick_random_new_stock(existing: set[str]) -> str | None:
    available = [t for t in NSE_UNIVERSE if t not in existing]
    return random.choice(available) if available else None


# ─────────────────────────────────────────────────────────────────────────────
# Market hours check
# ─────────────────────────────────────────────────────────────────────────────

def is_trading_hours() -> bool:
    """True if NSE is in normal trading session (9:15–15:30 IST, Mon–Fri)."""
    try:
        stocks = get_stocks()
        if stocks:
            r = _get(f"/api/stocks/{stocks[0]['ticker']}/live", timeout=8)
            if r.status_code == 200:
                return r.json().get("is_trading", False)
    except Exception:
        pass
    # Fallback: local IST time
    now = datetime.now(IST)
    return (
        now.weekday() < 5 and
        now.replace(hour=9, minute=15, second=0, microsecond=0) <= now <=
        now.replace(hour=15, minute=30, second=0, microsecond=0)
    )


def seconds_until_market_open() -> float:
    """Seconds until 9:10 IST on the next trading day (5-min early buffer)."""
    now = datetime.now(IST)
    target = now.replace(hour=9, minute=10, second=0, microsecond=0)
    if now >= target:
        # Move to next day
        target += timedelta(days=1)
    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return max(0.0, (target - now).total_seconds())


def seconds_to_next_bar(extra_buffer: int = 45) -> float:
    """
    Returns how long (seconds) to sleep until the NEXT 10-minute bar
    has had time to close and be available from yfinance.

    Logic: we wait until the next 10-min boundary + 10 minutes (for the
    predicted bar to accumulate) + a small buffer.
    """
    now = datetime.now(IST)
    m, s = now.minute, now.second
    # Seconds remaining in the current 10-min window
    secs_to_next_mark = (10 - m % 10) * 60 - s
    # After that mark, the bar we predicted needs another 10 min to form.
    # But actually at the mark, the PREVIOUS bar is complete, which is the one
    # our model already used. We wait until the next mark after THAT.
    return max(60, secs_to_next_mark + extra_buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Intraday API helpers
# ─────────────────────────────────────────────────────────────────────────────

def intraday_train(ticker: str) -> dict:
    """POST /api/stocks/{ticker}/intraday/train"""
    try:
        r = _post(f"/api/stocks/{ticker}/intraday/train", timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def intraday_predict(ticker: str) -> dict:
    """POST /api/stocks/{ticker}/intraday/predict"""
    try:
        r = _post(f"/api/stocks/{ticker}/intraday/predict", timeout=20)
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def intraday_record_actual(ticker: str, actual_price: float) -> dict:
    """POST /api/stocks/{ticker}/intraday/record-actual"""
    try:
        r = _post(
            f"/api/stocks/{ticker}/intraday/record-actual",
            json={"actual_price": actual_price},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception as e:
        return {"error": str(e)}


def intraday_replay(ticker: str, days_back: int = 5) -> dict:
    """POST /api/stocks/{ticker}/intraday/replay"""
    try:
        r = _post(
            f"/api/stocks/{ticker}/intraday/replay",
            params={"days_back": days_back},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _direction_str(direction: int | None) -> str:
    if direction is None:
        return "?"
    return _green("UP  ") if direction == 1 else _red("DOWN")


def print_prediction_table(predictions: dict[str, dict]) -> None:
    """Print a compact table of current intraday predictions."""
    print(f"\n  {'TICKER':<14}  {'DIR':>4}  {'CONF':>6}  {'PRICE':>10}  {'RANGE'}")
    print(f"  {'-'*14}  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*22}")
    for ticker, pred in predictions.items():
        if "error" in pred:
            print(f"  {ticker:<14}  {_dim('no model yet')}")
            continue
        d     = _direction_str(pred.get("direction"))
        conf  = f"{pred.get('confidence', 0) * 100:.0f}%"
        price = f"Rs.{pred.get('predicted_price', 0):,.2f}"
        lo    = pred.get("lower_band", 0)
        hi    = pred.get("upper_band", 0)
        rng   = f"Rs.{lo:,.0f}–{hi:,.0f}"
        nxt   = pred.get("next_time", "?")
        print(f"  {_bold(ticker):<20}  {d}  {conf:>6}  {price:>12}  {rng}  @{nxt}")
    print()


def print_accuracy_table(results: dict[str, dict]) -> None:
    """Print session accuracy after recording actuals."""
    print(f"\n  {'TICKER':<14}  {'RESULT':>8}  {'SESSION ACC':>12}  {'CHECKS':>7}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*12}  {'-'*7}")
    for ticker, res in results.items():
        correct = res.get("prediction_correct")
        session = res.get("session_accuracy", {})
        acc     = session.get("accuracy")
        total   = session.get("total", 0)
        verdict = _green("CORRECT") if correct else _red("WRONG  ") if correct is False else _dim("pending")
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "-"
        col     = _green if (acc or 0) >= 0.54 else _yellow if (acc or 0) >= 0.50 else _red
        print(f"  {ticker:<14}  {verdict}  {col(acc_str):>18}  {total:>7}")
    print()


def print_watchlist(stocks: list[dict]) -> None:
    separator("Watchlist")
    print(f"  {'TICKER':<14}  {'ACCURACY':>9}  {'LAST UPDATED'}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*20}")
    for s in stocks:
        acc = s.get("model_accuracy", 0)
        lu  = (s.get("last_updated") or "never")[:19]
        col = _green if acc >= 54 else _yellow if acc >= 51 else _red
        print(f"  {s['ticker']:<14}  {col(f'{acc:>8.1f}%')}  {_dim(lu)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MARKET OPEN: 10-minute intraday loop
# ─────────────────────────────────────────────────────────────────────────────

def run_intraday_loop(stocks: list[dict]) -> None:
    """
    Main intraday training loop.  Runs while the market is open:
      1. Train all intraday models on last 5 days' bars.
      2. Predict the next 10-minute bar for every stock.
      3. Wait for that bar to close (~10 min + buffer).
      4. Record actual prices, compute accuracy.
      5. Retrain on the new bar.
      6. Go to 2.
    Returns when the market closes.
    """
    tickers = [s["ticker"] for s in stocks]
    if not tickers:
        log("No stocks to track intraday.", "WARN")
        return

    separator("Intraday Loop — Market OPEN")
    log(f"Tracking {len(tickers)} stock(s) on 10-min bars.", "INTRA")
    log(f"Bar interval: {INTRADAY_BAR_SECONDS}s + {INTRADAY_BAR_BUFFER}s buffer.", "INTRA")

    # ── Step 1: initial train ────────────────────────────────────────────────
    log("Training intraday models on last 5 days' bars...", "INTRA")
    for ticker in tickers:
        result = intraday_train(ticker)
        if "error" in result:
            log(f"  {ticker}: {result['error']}", "WARN")
        else:
            log(f"  {ticker}: trained on {result.get('trained_bars', '?')} bars", "OK")

    # We store each iteration's predictions here so we can record actuals after sleeping
    last_predictions: dict[str, dict] = {}
    bar_number = 0

    while is_trading_hours():
        bar_number += 1
        separator(f"Intraday Bar {bar_number}  [{datetime.now(IST).strftime('%H:%M:%S')} IST]")

        # ── Step 2: make predictions ─────────────────────────────────────────
        log("Making next-bar predictions...", "INTRA")
        current_predictions: dict[str, dict] = {}
        for ticker in tickers:
            pred = intraday_predict(ticker)
            current_predictions[ticker] = pred
            if "error" in pred:
                log(f"  {ticker}: {pred.get('error', 'failed')}", "WARN")

        print_prediction_table(current_predictions)

        # ── Step 3: wait for bar to close ────────────────────────────────────
        wait_s = seconds_to_next_bar(extra_buffer=INTRADAY_BAR_BUFFER)
        log(f"Waiting {wait_s:.0f}s for next bar to close "
            f"(until ~{(datetime.now(IST) + timedelta(seconds=wait_s)).strftime('%H:%M:%S')} IST)...",
            "INTRA")

        # Countdown display
        start_wait = time.time()
        while time.time() - start_wait < wait_s:
            remaining = wait_s - (time.time() - start_wait)
            mins, secs = divmod(int(remaining), 60)
            print(f"\r  [waiting]  {mins:02d}m {secs:02d}s remaining...  ", end="", flush=True)
            time.sleep(5)
            if not is_trading_hours():
                print()
                log("Market closed during wait — exiting intraday loop.", "INTRA")
                return
        print()

        # ── Step 4: record actuals ───────────────────────────────────────────
        log("Recording actual prices...", "INTRA")
        accuracy_results: dict[str, dict] = {}
        for ticker in tickers:
            live = get_live_price(ticker)
            actual_price = live.get("price")
            if actual_price is None:
                log(f"  {ticker}: no live price available", "WARN")
                accuracy_results[ticker] = {}
                continue
            result = intraday_record_actual(ticker, actual_price)
            accuracy_results[ticker] = result

        print_accuracy_table(accuracy_results)

        # ── Step 5: retrain on updated bars ──────────────────────────────────
        log("Retraining intraday models on updated bars...", "INTRA")
        for ticker in tickers:
            result = intraday_train(ticker)
            if "error" in result:
                log(f"  {ticker}: retrain failed — {result['error']}", "WARN")
            else:
                log(f"  {ticker}: retrained on {result.get('trained_bars', '?')} bars", "OK")

    log("Market closed. Exiting intraday loop.", "INTRA")


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CLOSED: offline replay + overnight retrain
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_replay(stocks: list[dict]) -> None:
    """
    Run after market close:
      1. Replay last 5 days of 10-min bars for each stock (predict → check → retrain).
      2. Report replay accuracy — this is how the intraday model self-evaluates offline.
    """
    tickers = [s["ticker"] for s in stocks]
    if not tickers:
        return

    separator("Offline Intraday Replay")
    log(f"Replaying last 5 days of 10-min bars for {len(tickers)} stock(s).", "INTRA")
    log("Each bar: train on history → predict next → compare actual → advance.", "INTRA")

    for ticker in tickers:
        log(f"Replaying {ticker}...")
        result = intraday_replay(ticker, days_back=5)
        if "error" in result:
            log(f"  {ticker}: {result['error']}", "WARN")
            continue
        n      = result.get("total_predictions", 0)
        acc    = result.get("replay_accuracy")
        days   = result.get("days_replayed", 0)
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        col    = _green if (acc or 0) >= 0.54 else _yellow if (acc or 0) >= 0.50 else _red
        log(f"  {ticker}: {days}d replayed, {n} predictions, replay accuracy: {col(acc_str)}", "OK")


def refresh_all_stocks(stocks: list[dict], cycle: int) -> None:
    """
    Trigger full overnight Optuna retrains for all tracked stocks.
    This is the main model-improvement step that runs every night.
    """
    if not stocks:
        return

    separator("Overnight Optuna Retrain")
    log(f"Refreshing {len(stocks)} stock(s) with Optuna hyperparameter search...")

    for i, s in enumerate(stocks, 1):
        ticker       = s["ticker"]
        prev_updated = s.get("last_updated")

        log(f"[{i}/{len(stocks)}] Triggering refresh for {ticker}...")
        if not refresh_stock(ticker):
            log(f"Could not trigger refresh for {ticker}, skipping.", "WARN")
            continue

        done = wait_for_pipeline(ticker, prev_updated)
        if done:
            fresh = get_stocks()
            for fs in fresh:
                if fs["ticker"] == ticker:
                    acc = fs.get("model_accuracy", 0)
                    log(f"{ticker} done — walk-forward accuracy: {acc:.1f}%", "OK")
                    break
        else:
            log(f"{ticker} pipeline timed out.", "WARN")

        if i < len(stocks):
            log(f"Sleeping {SLEEP_BETWEEN_STOCKS}s before next stock...")
            time.sleep(SLEEP_BETWEEN_STOCKS)

    # Possibly add a new stock
    if cycle % ADD_NEW_EVERY_N_CYCLES == 0:
        current_stocks  = get_stocks()
        current_tickers = {s["ticker"] for s in current_stocks}

        if len(current_stocks) >= MAX_STOCKS:
            log(f"At {MAX_STOCKS} stock cap — not adding more. Raise MAX_STOCKS to expand.")
        else:
            new_ticker = pick_random_new_stock(current_tickers)
            if new_ticker:
                log(f"Adding random new stock: {_bold(new_ticker)}")
                if add_stock(new_ticker):
                    done = wait_for_pipeline(new_ticker, previous_updated=None)
                    if done:
                        fresh_all = get_stocks()
                        for fs in fresh_all:
                            if fs["ticker"] == new_ticker:
                                log(f"{new_ticker} initial training done — "
                                    f"acc: {fs.get('model_accuracy', 0):.1f}%", "OK")
                                break
                else:
                    log(f"Failed to add {new_ticker}.", "ERROR")
            else:
                log("All NSE_UNIVERSE stocks already tracked.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    separator("NSE Auto-Trainer")
    log(f"Platform: {platform.system()} {platform.machine()}")
    log(f"Backend:  {BACKEND_URL}")
    log(f"Intraday: {INTRADAY_BAR_SECONDS}s bars, {INTRADAY_BAR_BUFFER}s buffer")
    log(f"Overnight: add new stock every {ADD_NEW_EVERY_N_CYCLES} cycles, cap {MAX_STOCKS}")
    separator()

    # Health check
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        r.raise_for_status()
        log("Backend is healthy.", "OK")
    except Exception:
        log("Backend not reachable. Start the backend first (uvicorn main:app).", "ERROR")
        sys.exit(1)

    overnight_cycle = 0

    while True:
        stocks = get_stocks()
        if not stocks:
            log("No stocks in watchlist yet — add one via the web UI.", "WARN")
            time.sleep(30)
            continue

        print_watchlist(stocks)

        if is_trading_hours():
            # ── MARKET OPEN: intraday 10-min loop ──────────────────────────
            log("NSE market is OPEN. Entering intraday loop.", "LIVE")
            run_intraday_loop(stocks)
            # After market closes, refresh the stock list
            stocks = get_stocks()

        else:
            # ── MARKET CLOSED: replay + overnight retrain ───────────────────
            overnight_cycle += 1
            log(f"NSE market is CLOSED. Overnight cycle {overnight_cycle}.", "INFO")

            run_offline_replay(stocks)
            refresh_all_stocks(stocks, overnight_cycle)

            # Sleep until 9:10 IST
            wait_s = seconds_until_market_open()
            wake   = datetime.now(IST) + timedelta(seconds=wait_s)
            log(f"All done. Sleeping {wait_s / 3600:.1f}h until "
                f"{wake.strftime('%Y-%m-%d %H:%M')} IST when market opens...")
            separator()
            time.sleep(wait_s)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        separator("Stopped")
        log("Auto-trainer stopped (Ctrl+C). All models and Optuna studies are saved.", "OK")
        sys.exit(0)
