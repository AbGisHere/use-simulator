#!/usr/bin/env python3
"""
auto_trainer.py — Continuous background trainer for NSE Simulator
==================================================================
Keeps every stock in your watchlist trained and improving via Optuna.
Also randomly explores new stocks from a curated NSE universe.
During NSE trading hours it enters a "live mode" — tracking predictions
vs real prices every N seconds instead of triggering full retrains.

Cross-platform: works on macOS, Linux, and Windows.

Usage:
    # macOS / Linux:
    cd backend
    source .venv/bin/activate
    python auto_trainer.py

    # Windows (PowerShell):
    cd backend
    .venv\\Scripts\\Activate.ps1
    python auto_trainer.py

Options: edit the CONFIG block below.
"""

import os
import platform
import random
import sys
import time
from datetime import datetime

import requests

# ── Detect platform ───────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — tweak these without touching anything else
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL             = "http://localhost:8000"
SLEEP_BETWEEN_STOCKS    = 30          # seconds gap between pipeline runs
ADD_NEW_EVERY_N_CYCLES  = 2           # add a random stock every N full cycles
MAX_STOCKS              = 20          # cap total tracked stocks
POLL_TIMEOUT            = 600         # max seconds to wait for a pipeline
POLL_INTERVAL           = 15          # seconds between pipeline status polls

# Live mode — used during NSE trading hours instead of triggering retrains
LIVE_POLL_INTERVAL      = 10          # seconds between live price polls (min 5)
LIVE_TRACK_DURATION     = 60 * 60     # stay in live mode for this many seconds
                                      # (then trigger one full retrain)
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
# Terminal helpers (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Return True if the terminal likely supports ANSI colour codes."""
    if IS_WINDOWS:
        # Windows Terminal / ConEmu / VS Code support ANSI; cmd.exe does not
        return os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") or \
               os.environ.get("COLORTERM") or False
    return True  # macOS / Linux almost always support colour


_USE_COLOR = _supports_color()

def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _blue(t):   return _c(t, "34")
def _bold(t):   return _c(t, "1")
def _dim(t):    return _c(t, "2")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    icons = {"INFO": "·", "OK": "✓", "WARN": "!", "ERROR": "✗", "LIVE": "◉"}
    icon  = icons.get(level, "·")
    color = {"OK": _green, "WARN": _yellow, "ERROR": _red, "LIVE": _blue}.get(level, lambda x: x)
    print(f"[{_dim(_ts())}] {color(icon + '  ' + msg)}", flush=True)


def separator(title: str = "") -> None:
    width = 70
    line  = "─" * width
    if title:
        pad  = (width - len(title) - 2) // 2
        line = "─" * pad + f" {title} " + "─" * pad
    print(f"\n{_dim(line)}\n", flush=True)


def print_watchlist_summary(stocks: list[dict]) -> None:
    separator("Watchlist")
    print(f"  {'TICKER':<14} {'ACCURACY':>9}  {'LAST UPDATED'}")
    print(f"  {'─'*14} {'─'*9}  {'─'*20}")
    for s in stocks:
        acc = s.get("model_accuracy", 0)
        lu  = (s.get("last_updated") or "never")[:19]
        col = _green if acc >= 54 else _yellow if acc >= 51 else _red
        print(f"  {s['ticker']:<14} {col(f'{acc:>8.1f}%')}  {_dim(lu)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(path: str, **kw):
    return requests.get(f"{BACKEND_URL}{path}", timeout=kw.pop("timeout", 10), **kw)


def _post(path: str, **kw):
    return requests.post(f"{BACKEND_URL}{path}", timeout=kw.pop("timeout", 10), **kw)


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


def wait_for_pipeline(ticker: str, prev_updated: str | None) -> bool:
    deadline = time.time() + POLL_TIMEOUT
    dots = 0
    while time.time() < deadline:
        current = get_last_updated(ticker)
        if current and current != prev_updated:
            print()
            return True
        dots = (dots + 1) % 4
        # Cross-platform carriage-return trick
        print(f"\r[{_ts()}] {_dim('Waiting for ' + ticker + ' pipeline ' + '.' * dots + '   ')}", end="", flush=True)
        time.sleep(POLL_INTERVAL)
    print()
    log(f"Timed out waiting for {ticker} after {POLL_TIMEOUT}s", "WARN")
    return False


def pick_random_new_stock(existing: set[str]) -> str | None:
    available = [t for t in NSE_UNIVERSE if t not in existing]
    return random.choice(available) if available else None


# ─────────────────────────────────────────────────────────────────────────────
# Market hours check (via backend)
# ─────────────────────────────────────────────────────────────────────────────

def is_trading_hours() -> bool:
    """Check if NSE is in normal trading session via the backend live endpoint."""
    try:
        stocks = get_stocks()
        if not stocks:
            return False
        ticker = stocks[0]["ticker"]
        r = _get(f"/api/stocks/{ticker}/live", timeout=8)
        if r.status_code == 200:
            return r.json().get("is_trading", False)
    except Exception:
        pass
    # Fallback: check local time (IST = UTC+5:30)
    import datetime as dt
    utc_now   = dt.datetime.utcnow()
    ist_now   = utc_now + dt.timedelta(hours=5, minutes=30)
    weekday   = ist_now.weekday()
    t         = ist_now.time()
    return (weekday < 5) and (dt.time(9, 15) <= t <= dt.time(15, 30))


# ─────────────────────────────────────────────────────────────────────────────
# Live tracking mode (during trading hours)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_tracking_session(stocks: list[dict], duration_seconds: int) -> None:
    """
    During trading hours: poll live prices for all tracked stocks and show
    how the model's predictions are tracking actual market movement.
    Runs for `duration_seconds` then returns (to trigger retraining).
    """
    separator("Live Tracking Mode")
    log(f"NSE market is OPEN. Tracking {len(stocks)} stocks for {duration_seconds // 60}min before retraining.", "LIVE")
    log(f"Polling every {LIVE_POLL_INTERVAL}s. Press Ctrl+C to stop.", "LIVE")

    end_time     = time.time() + duration_seconds
    correct_hits = {s["ticker"]: 0 for s in stocks}
    total_hits   = {s["ticker"]: 0 for s in stocks}

    while time.time() < end_time:
        print()
        remaining = int(end_time - time.time())
        print(f"  {_dim(f'[{_ts()}]')}  {_blue('◉ LIVE')}  — {remaining // 60}m {remaining % 60}s until retrain")
        print()

        for s in stocks:
            ticker = s["ticker"]
            try:
                r = _get(f"/api/stocks/{ticker}/live", timeout=8)
                if r.status_code != 200:
                    continue
                d = r.json()

                price      = d.get("price")
                change_pct = d.get("change_pct")
                mc         = d.get("model_comparison")
                status     = d.get("market_status", "?")

                if price is None:
                    print(f"  {ticker:<14}  {_dim('No price available')}")
                    continue

                # Direction indicator
                arrow = _green("↑") if (change_pct or 0) >= 0 else _red("↓")
                chg   = f"({'+' if (change_pct or 0) >= 0 else ''}{change_pct:.2f}%)" if change_pct else ""

                # Model accuracy indicator
                accuracy_str = ""
                if mc:
                    total_hits[ticker] += 1
                    if mc["prediction_correct"]:
                        correct_hits[ticker] += 1
                    session_acc = correct_hits[ticker] / total_hits[ticker] * 100
                    verdict     = _green("✓ correct") if mc["prediction_correct"] else _yellow("✗ diverging")
                    accuracy_str = f"  model {verdict}  session acc: {session_acc:.0f}%"

                print(f"  {_bold(ticker):<20}  {arrow} ₹{price:>9,.2f}  {_dim(chg):<12}{accuracy_str}")

            except Exception as exc:
                print(f"  {ticker:<14}  {_dim(f'Error: {exc}')}")

        time.sleep(LIVE_POLL_INTERVAL)

    separator()
    log("Live tracking session ended — triggering retrains now.", "LIVE")

    # Print session accuracy summary
    for ticker in correct_hits:
        if total_hits[ticker] > 0:
            acc = correct_hits[ticker] / total_hits[ticker] * 100
            log(f"{ticker}: {total_hits[ticker]} checks, model was correct {acc:.0f}% of the time this session", "INFO")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    separator("NSE Auto-Trainer")
    log(f"Platform: {platform.system()} {platform.machine()}")
    log(f"Backend:  {BACKEND_URL}")
    log(f"Gap between stocks: {SLEEP_BETWEEN_STOCKS}s")
    log(f"Add new stock every {ADD_NEW_EVERY_N_CYCLES} cycle(s), up to {MAX_STOCKS} stocks")
    log(f"Live tracking: {LIVE_POLL_INTERVAL}s poll, {LIVE_TRACK_DURATION // 60}min sessions during market hours")
    separator()

    # Health check
    try:
        r = _get("/api/health")
        r.raise_for_status()
        log("Backend is healthy.", "OK")
    except Exception:
        log("Backend is not reachable. Make sure the server is running first.", "ERROR")
        sys.exit(1)

    cycle = 0

    while True:
        cycle += 1
        separator(f"Cycle {cycle}")

        stocks = get_stocks()
        if not stocks:
            log("No stocks in watchlist — will add one.", "WARN")

        print_watchlist_summary(stocks)

        # ── During trading hours: live tracking first ──────────────────────
        if is_trading_hours():
            log("NSE market is OPEN.", "LIVE")
            run_live_tracking_session(stocks, duration_seconds=LIVE_TRACK_DURATION)
            # After the live session, refresh all stocks to incorporate today's data
            stocks = get_stocks()  # refresh list

        # ── A. Refresh all existing stocks ─────────────────────────────────
        if stocks:
            log(f"Refreshing {len(stocks)} stock(s)...")
            for i, s in enumerate(stocks, 1):
                ticker      = s["ticker"]
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
                            log(f"{ticker} done  walk-forward accuracy: {acc:.1f}%", "OK")
                            break
                else:
                    log(f"{ticker} pipeline timed out.", "WARN")

                if i < len(stocks):
                    log(f"Sleeping {SLEEP_BETWEEN_STOCKS}s...")
                    time.sleep(SLEEP_BETWEEN_STOCKS)

        # ── B. Possibly add a random new stock ─────────────────────────────
        if cycle % ADD_NEW_EVERY_N_CYCLES == 0:
            current_stocks  = get_stocks()
            current_tickers = {s["ticker"] for s in current_stocks}

            if len(current_stocks) >= MAX_STOCKS:
                log(f"At {MAX_STOCKS} stocks — not adding more (raise MAX_STOCKS to expand).")
            else:
                new_ticker = pick_random_new_stock(current_tickers)
                if new_ticker:
                    log(f"Adding random stock: {_bold(new_ticker)}")
                    if add_stock(new_ticker):
                        done = wait_for_pipeline(new_ticker, previous_updated=None)
                        if done:
                            fresh_all = get_stocks()
                            for fs in fresh_all:
                                if fs["ticker"] == new_ticker:
                                    log(f"{new_ticker} initial training done  acc: {fs.get('model_accuracy', 0):.1f}%", "OK")
                                    break
                    else:
                        log(f"Failed to add {new_ticker}.", "ERROR")
                else:
                    log("All NSE_UNIVERSE stocks already tracked.")

        separator()
        log(f"Cycle {cycle} complete. Starting cycle {cycle + 1}...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        separator("Stopped")
        log("Auto-trainer stopped (Ctrl+C). All models and Optuna studies are saved.", "OK")
        sys.exit(0)
