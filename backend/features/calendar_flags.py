import logging
from datetime import date, datetime, timedelta

import pandas as pd

from config import IST

logger = logging.getLogger(__name__)

# ── RBI MPC Meeting Dates (hardcoded for 2023-2025) ─────────────────────────
# Source: RBI official calendar - bi-monthly meetings
RBI_MEETING_DATES: list[date] = [
    # 2023
    date(2023, 2, 8), date(2023, 4, 6), date(2023, 6, 8),
    date(2023, 8, 10), date(2023, 10, 6), date(2023, 12, 8),
    # 2024
    date(2024, 2, 8), date(2024, 4, 5), date(2024, 6, 7),
    date(2024, 8, 8), date(2024, 10, 9), date(2024, 12, 6),
    # 2025
    date(2025, 2, 7), date(2025, 4, 9), date(2025, 6, 6),
    date(2025, 8, 8), date(2025, 10, 8), date(2025, 12, 5),
    # 2026
    date(2026, 2, 6), date(2026, 4, 9),
]


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of a given month (NSE F&O expiry day)."""
    # Start from the last day and walk backwards
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    days_back = (last_day.weekday() - 3) % 7  # Thursday = 3
    return last_day - timedelta(days=days_back)


def _get_fo_expiry_dates(start_year: int, end_year: int) -> list[date]:
    """Generate all monthly F&O expiry dates (last Thursday) for a year range."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_last_thursday_of_month(year, month))
    return dates


def _quarter_end_weeks(years: list[int]) -> list[date]:
    """
    Return dates in the last week of March, June, September, December.
    Quarter-end months: 3, 6, 9, 12. Last week = days 24-31.
    """
    dates = []
    for year in years:
        for month in [3, 6, 9, 12]:
            if month in [3, 5, 7, 8, 10, 12]:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 28
            for day in range(last_day - 6, last_day + 1):
                try:
                    dates.append(date(year, month, day))
                except ValueError:
                    pass
    return dates


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary calendar feature columns to a trading day DataFrame.

    Expected: df.index is a DatetimeIndex (IST-aware).
    Adds columns:
        is_rbi_meeting_week, is_budget_day, is_fo_expiry,
        is_quarter_end, days_to_next_rbi
    """
    result = df.copy()

    # Get the date range
    start_year = result.index.min().year
    end_year = result.index.max().year
    years = list(range(start_year, end_year + 1))

    fo_expiry_dates = set(_get_fo_expiry_dates(start_year - 1, end_year + 1))
    quarter_end_dates = set(_quarter_end_weeks(years))
    rbi_dates = sorted(RBI_MEETING_DATES)

    # Pre-compute RBI meeting weeks
    rbi_weeks: set[tuple[int, int]] = set()
    for rbi_date in rbi_dates:
        # Include the entire ISO week (Mon-Sun)
        for delta in range(-3, 4):
            week_day = rbi_date + timedelta(days=delta)
            rbi_weeks.add((week_day.year, week_day.isocalendar()[1]))

    is_rbi_week = []
    is_budget = []
    is_fo_expiry_flag = []
    is_qend = []
    days_to_rbi = []

    future_rbi = [d for d in rbi_dates if d >= date.today()]

    for idx_dt in result.index:
        d = idx_dt.date() if hasattr(idx_dt, "date") else idx_dt

        # RBI meeting week
        iso_week = d.isocalendar()[1]
        is_rbi_week.append(int((d.year, iso_week) in rbi_weeks))

        # Budget day: February 1
        is_budget.append(int(d.month == 2 and d.day == 1))

        # F&O expiry
        is_fo_expiry_flag.append(int(d in fo_expiry_dates))

        # Quarter end
        is_qend.append(int(d in quarter_end_dates))

        # Days to next RBI meeting
        next_rbi_candidates = [rd for rd in rbi_dates if rd >= d]
        if next_rbi_candidates:
            next_rbi = next_rbi_candidates[0]
            delta_days = (next_rbi - d).days
        else:
            delta_days = 999  # far future
        days_to_rbi.append(delta_days)

    result["is_rbi_meeting_week"] = is_rbi_week
    result["is_budget_day"] = is_budget
    result["is_fo_expiry"] = is_fo_expiry_flag
    result["is_quarter_end"] = is_qend
    result["days_to_next_rbi"] = days_to_rbi

    logger.info("Added calendar flags to %d rows", len(result))
    return result
