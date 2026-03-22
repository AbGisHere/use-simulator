import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from model.signals import generate_signals

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000.0  # ₹1 lakh starting capital
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade (brokerage + STT approximation)


def run_backtest(prediction_df: pd.DataFrame) -> dict[str, Any]:
    """
    Simulate trading based on model signals from day 1 to today.

    Strategy:
        - Enter long when BUY signal on day T (at next day's open ≈ close)
        - Exit when SELL signal or no signal the next day
        - Single position at a time (fully invested or cash)

    Returns a dict with:
        - portfolio_series: pd.DataFrame with date, portfolio_value, benchmark_value
        - stats: cumulative_return, sharpe_ratio, max_drawdown, win_rate, n_trades
        - benchmark_return: buy-and-hold return
    """
    if prediction_df.empty:
        return _empty_result()

    # Add signals
    df = generate_signals(prediction_df.copy())
    df = df.sort_values("date").reset_index(drop=True)

    # ── Portfolio simulation ──────────────────────────────────────────────────
    portfolio_value = INITIAL_CAPITAL
    benchmark_value = INITIAL_CAPITAL
    in_position = False
    entry_price = 0.0
    trade_returns: list[float] = []
    portfolio_series: list[dict[str, Any]] = []

    first_price = df["actual_price"].iloc[0]
    benchmark_shares = benchmark_value / first_price  # buy and hold

    for i, row in df.iterrows():
        price = row["actual_price"]
        signal = row.get("signal", "hold")

        # Benchmark: always holds
        current_benchmark = benchmark_shares * price

        if not in_position:
            if signal == "buy":
                # Enter long: pay transaction cost
                entry_price = price * (1 + TRANSACTION_COST_PCT)
                in_position = True

            # Portfolio stays in cash (no return on cash for simplicity)
            current_portfolio = portfolio_value

        else:  # in position
            current_value = (portfolio_value / entry_price) * price

            if signal == "sell" or (i == len(df) - 1):
                # Exit: pay transaction cost
                exit_price = price * (1 - TRANSACTION_COST_PCT)
                trade_return = (exit_price - entry_price) / entry_price
                portfolio_value = (portfolio_value / entry_price) * exit_price
                trade_returns.append(trade_return)
                in_position = False
                entry_price = 0.0

            current_portfolio = current_value

        portfolio_series.append(
            {
                "date": row["date"],
                "portfolio_value": current_portfolio,
                "benchmark_value": current_benchmark,
            }
        )

    port_df = pd.DataFrame(portfolio_series)

    # ── Statistics ────────────────────────────────────────────────────────────
    final_portfolio = port_df["portfolio_value"].iloc[-1]
    final_benchmark = port_df["benchmark_value"].iloc[-1]

    cumulative_return = (final_portfolio - INITIAL_CAPITAL) / INITIAL_CAPITAL
    benchmark_return = (final_benchmark - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Daily returns for Sharpe
    port_daily_returns = port_df["portfolio_value"].pct_change().dropna()
    sharpe = _sharpe_ratio(port_daily_returns)

    # Max drawdown
    max_drawdown = _max_drawdown(port_df["portfolio_value"])

    # Win rate
    n_trades = len(trade_returns)
    win_rate = sum(1 for r in trade_returns if r > 0) / n_trades if n_trades > 0 else 0.0

    stats = {
        "cumulative_return": round(cumulative_return * 100, 2),
        "benchmark_return": round(benchmark_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "n_trades": n_trades,
        "final_portfolio_value": round(final_portfolio, 2),
        "final_benchmark_value": round(final_benchmark, 2),
        "alpha": round((cumulative_return - benchmark_return) * 100, 2),
    }

    logger.info("Backtest stats for ticker: %s", stats)
    return {"portfolio_series": port_df, "stats": stats}


def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Annualised Sharpe ratio (assume 252 trading days; Indian risk-free ≈ 6.5%)."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    return math.sqrt(252) * excess.mean() / excess.std()


def _max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown from a portfolio value series."""
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return abs(drawdown.min())


def _empty_result() -> dict[str, Any]:
    return {
        "portfolio_series": pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"]),
        "stats": {
            "cumulative_return": 0.0,
            "benchmark_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "final_portfolio_value": INITIAL_CAPITAL,
            "final_benchmark_value": INITIAL_CAPITAL,
            "alpha": 0.0,
        },
    }
