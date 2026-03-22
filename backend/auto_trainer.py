#!/usr/bin/env python3
"""
auto_trainer.py — Continuous background trainer for NSE Simulator
==================================================================
Keeps every stock in your watchlist trained and improving via Optuna.
Also randomly explores new stocks from a curated NSE universe.

Usage:
    cd backend
    source .venv/bin/activate
    python auto_trainer.py

Options (edit the CONFIG block below):
    BACKEND_URL         URL of your running backend  (default: localhost:8000)
    SLEEP_BETWEEN_STOCKS  Seconds to wait between two pipeline runs (default: 30)
    ADD_NEW_EVERY_N_CYCLES  Add a random new stock every N full cycles (default: 2)
    MAX_STOCKS          Stop adding new stocks above this count   (default: 20)
    POLL_TIMEOUT        Max seconds to wait for a pipeline to finish (default: 600)
    POLL_INTERVAL       How often to check if pipeline is done (seconds, default: 15)

Press Ctrl+C to stop gracefully.
"""

import random
import sys
import time
from datetime import datetime, timezone

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — tweak these without touching anything else
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL         = "http://localhost:8000"
SLEEP_BETWEEN_STOCKS = 30          # seconds between consecutive pipeline runs
ADD_NEW_EVERY_N_CYCLES = 2         # add a random stock every N full cycles
MAX_STOCKS          = 20           # don't add beyond this many tracked stocks
POLL_TIMEOUT        = 600          # seconds before giving up waiting for pipeline
POLL_INTERVAL       = 15           # seconds between status polls

# ─────────────────────────────────────────────────────────────────────────────
# NSE universe to explore (these are all in your sector_taxonomy.py)
# The trainer will randomly pick from ones NOT yet in your watchlist.
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    """Current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    symbols = {"INFO": "·", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "HEAD": "━"}
    sym = symbols.get(level, "·")
    print(f"[{_ts()}] {sym}  {msg}", flush=True)


def separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'━' * pad} {title} {'━' * pad}\n", flush=True)
    else:
        print(f"\n{'━' * width}\n", flush=True)


def get_stocks() -> list[dict]:
    """Fetch the current watchlist from the backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/stocks", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Could not reach backend: {e}", "ERROR")
        return []


def get_last_updated(ticker: str) -> str | None:
    """Return the last_updated timestamp string for a ticker."""
    stocks = get_stocks()
    for s in stocks:
        if s["ticker"] == ticker:
            return s.get("last_updated")
    return None


def refresh_stock(ticker: str) -> bool:
    """POST /api/stocks/{ticker}/refresh — starts pipeline as background task."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/stocks/{ticker}/refresh", timeout=10)
        if r.status_code in (200, 202):
            return True
        log(f"Refresh returned {r.status_code} for {ticker}", "WARN")
        return False
    except Exception as e:
        log(f"Refresh request failed for {ticker}: {e}", "ERROR")
        return False


def add_stock(ticker: str) -> bool:
    """POST /api/stocks — add a new ticker to the watchlist."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/stocks",
            json={"ticker": ticker},
            timeout=10,
        )
        if r.status_code in (200, 201, 202):
            return True
        log(f"Add returned {r.status_code} for {ticker}: {r.text[:120]}", "WARN")
        return False
    except Exception as e:
        log(f"Add request failed for {ticker}: {e}", "ERROR")
        return False


def wait_for_pipeline(ticker: str, previous_updated: str | None) -> bool:
    """
    Poll until last_updated changes (meaning the pipeline finished).
    Returns True on success, False on timeout.
    """
    deadline = time.time() + POLL_TIMEOUT
    dots = 0
    while time.time() < deadline:
        current = get_last_updated(ticker)
        if current and current != previous_updated:
            return True
        # Animated waiting indicator
        dots = (dots + 1) % 4
        print(f"\r[{_ts()}] ⏳  Waiting for {ticker} pipeline {'.' * dots}   ", end="", flush=True)
        time.sleep(POLL_INTERVAL)
    print()  # newline after the animated dots
    log(f"Timed out waiting for {ticker} pipeline after {POLL_TIMEOUT}s", "WARN")
    return False


def pick_random_new_stock(existing_tickers: set[str]) -> str | None:
    """Pick a random ticker from NSE_UNIVERSE not already tracked."""
    available = [t for t in NSE_UNIVERSE if t not in existing_tickers]
    if not available:
        return None
    return random.choice(available)


def print_watchlist_summary(stocks: list[dict]) -> None:
    """Print a neat table of current stocks and their accuracy."""
    separator("Current Watchlist")
    print(f"  {'TICKER':<14} {'ACCURACY':>9}  {'LAST UPDATED'}")
    print(f"  {'─'*14} {'─'*9}  {'─'*20}")
    for s in stocks:
        acc = s.get("model_accuracy", 0)
        lu  = (s.get("last_updated") or "never")[:19]
        print(f"  {s['ticker']:<14} {acc:>8.1f}%  {lu}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    separator("NSE Auto-Trainer")
    log("Starting continuous training loop. Press Ctrl+C to stop.")
    log(f"Backend: {BACKEND_URL}")
    log(f"Sleep between stocks: {SLEEP_BETWEEN_STOCKS}s")
    log(f"Add new stock every {ADD_NEW_EVERY_N_CYCLES} cycle(s), up to {MAX_STOCKS} stocks")
    separator()

    # Quick health check
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        r.raise_for_status()
        log("Backend is healthy.", "OK")
    except Exception:
        log(
            "Backend is not reachable. Make sure ./start.sh is running, then retry.",
            "ERROR",
        )
        sys.exit(1)

    cycle = 0

    while True:
        cycle += 1
        separator(f"Cycle {cycle}")

        stocks = get_stocks()
        if not stocks:
            log("No stocks in watchlist yet — will try adding one.", "WARN")

        existing_tickers = {s["ticker"] for s in stocks}
        print_watchlist_summary(stocks)

        # ── A. Refresh all existing stocks ──────────────────────────────────
        if stocks:
            log(f"Refreshing {len(stocks)} existing stock(s)…")
            for i, s in enumerate(stocks, 1):
                ticker = s["ticker"]
                prev_updated = s.get("last_updated")

                log(f"[{i}/{len(stocks)}] Triggering refresh for {ticker}…")
                ok = refresh_stock(ticker)
                if not ok:
                    log(f"Could not trigger refresh for {ticker}, skipping.", "WARN")
                    continue

                log(f"Pipeline running for {ticker} — waiting for completion…")
                done = wait_for_pipeline(ticker, prev_updated)

                if done:
                    # Fetch fresh accuracy after training
                    fresh = get_stocks()
                    for fs in fresh:
                        if fs["ticker"] == ticker:
                            acc = fs.get("model_accuracy", 0)
                            log(
                                f"{ticker} done ✓  walk-forward accuracy: {acc:.1f}%",
                                "OK",
                            )
                            break
                else:
                    log(f"{ticker} pipeline did not finish in time — moving on.", "WARN")

                if i < len(stocks):
                    log(f"Sleeping {SLEEP_BETWEEN_STOCKS}s before next stock…")
                    time.sleep(SLEEP_BETWEEN_STOCKS)

        # ── B. Possibly add a new random stock ───────────────────────────────
        if cycle % ADD_NEW_EVERY_N_CYCLES == 0:
            current_stocks = get_stocks()
            if len(current_stocks) >= MAX_STOCKS:
                log(
                    f"Already at {MAX_STOCKS} stocks — not adding more. "
                    f"Raise MAX_STOCKS in the config if you want more.",
                )
            else:
                current_tickers = {s["ticker"] for s in current_stocks}
                new_ticker = pick_random_new_stock(current_tickers)
                if new_ticker:
                    log(f"Adding new random stock: {new_ticker} 🎲")
                    ok = add_stock(new_ticker)
                    if ok:
                        log(f"Added {new_ticker} — waiting for initial pipeline…")
                        # Wait until it appears in the list with a last_updated
                        done = wait_for_pipeline(new_ticker, previous_updated=None)
                        if done:
                            fresh_all = get_stocks()
                            for fs in fresh_all:
                                if fs["ticker"] == new_ticker:
                                    acc = fs.get("model_accuracy", 0)
                                    log(
                                        f"{new_ticker} initial training done ✓  accuracy: {acc:.1f}%",
                                        "OK",
                                    )
                                    break
                        else:
                            log(f"{new_ticker} initial pipeline timed out.", "WARN")
                    else:
                        log(f"Failed to add {new_ticker}.", "ERROR")
                else:
                    log("All stocks in NSE_UNIVERSE are already tracked — nothing to add.")

        separator()
        log(f"Cycle {cycle} complete. Starting cycle {cycle + 1} immediately…")
        # No extra sleep here — the per-stock sleeps already pace things out.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        separator("Stopped")
        log("Auto-trainer stopped by user (Ctrl+C). Models and Optuna studies are saved.", "OK")
        sys.exit(0)
