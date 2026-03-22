import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import FINBERT_MODEL, IST, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE

logger = logging.getLogger(__name__)

# ── FinBERT model (singleton, loaded once) ───────────────────────────────────
_finbert_pipeline = None


def _get_finbert() -> Any:
    """Lazy-load FinBERT pipeline. Downloads from HuggingFace on first call."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Loading FinBERT model from HuggingFace (%s). This may take a moment...", FINBERT_MODEL)
        device = 0 if torch.cuda.is_available() else -1
        _finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT model loaded successfully")
    return _finbert_pipeline


def score_text(text: str) -> dict[str, float]:
    """
    Score a single text using FinBERT.

    Returns: {label: 'positive'|'negative'|'neutral', score: float, numeric: float (-1 to +1)}
    """
    if not text or len(text.strip()) < 10:
        return {"label": "neutral", "score": 0.5, "numeric": 0.0}

    try:
        pipe = _get_finbert()
        # Truncate text to avoid token limit issues
        truncated = text[:2000]
        result = pipe(truncated)[0]
        label = result["label"].lower()
        confidence = result["score"]

        # Map to numeric: positive=+conf, negative=-conf, neutral=0
        if label == "positive":
            numeric = confidence
        elif label == "negative":
            numeric = -confidence
        else:
            numeric = 0.0

        return {"label": label, "score": confidence, "numeric": numeric}
    except Exception as exc:
        logger.warning("FinBERT scoring error: %s", exc)
        return {"label": "neutral", "score": 0.5, "numeric": 0.0}


def score_news_items(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score a list of news items using FinBERT.
    Each item must have 'headline' and optionally 'body'.
    Adds 'sentiment_label', 'sentiment_score', 'sentiment_numeric' to each item.
    """
    scored = []
    for item in news_items:
        text = item.get("headline", "") + " " + (item.get("body", "") or "")[:500]
        sentiment = score_text(text)
        item = dict(item)
        item["sentiment_label"] = sentiment["label"]
        item["sentiment_score"] = sentiment["score"]
        item["sentiment_numeric"] = sentiment["numeric"]
        scored.append(item)
    return scored


def _get_trading_day(dt: datetime, price_df: pd.DataFrame) -> datetime:
    """
    Determine which trading day a news item belongs to.

    Rule: News published after market close (3:30 PM IST) belongs to the next trading day.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)

    market_close = dt.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)

    if dt > market_close:
        # Belongs to next trading day
        candidate = dt.date() + timedelta(days=1)
    else:
        candidate = dt.date()

    # Find the next actual trading day in price_df
    trading_dates = [d.date() for d in price_df.index.to_pydatetime()]

    # Walk forward until we find a trading day
    for i in range(7):  # max 7 days (handles weekends/holidays)
        check_date = candidate + timedelta(days=i)
        if check_date in trading_dates:
            # Return as IST datetime at noon (neutral intra-day time)
            return datetime.combine(check_date, datetime.min.time().replace(hour=12)).replace(tzinfo=IST)

    # If no match found, return the original date
    return dt


# ── Match-type weights for sector-level news ─────────────────────────────────
# Company-direct news gets full weight. Sector/proxy news (PVR example: Bollywood
# box-office news affects PVR even without mentioning it) get reduced weights
# so they contribute meaningfully without overpowering direct company coverage.
MATCH_TYPE_WEIGHTS: dict[str, float] = {
    "company": 1.0,    # Direct company mention — full weight
    "sector": 0.6,     # Industry/sector news — medium influence
    "proxy": 0.4,      # Thematic proxy events — lighter influence
}


def build_daily_sentiment(
    news_items: list[dict[str, Any]],
    price_df: pd.DataFrame,
    nse_announcement_weight: float = 2.0,
) -> pd.Series:
    """
    Aggregate news sentiment into a single daily score per trading day.

    Weighting rules (applied multiplicatively):
    - match_type='company' → 1.0  (direct company news, default)
    - match_type='sector'  → 0.6  (sector/industry news from taxonomy)
    - match_type='proxy'   → 0.4  (commodity/policy proxy events)
    - NSE official announcements  → 2x on top of above
    - Reddit posts         → log10(upvotes+1) weight
    - News after 3:30 PM IST → assigned to next trading day

    Returns a pd.Series indexed by date with sentiment scores (-1 to +1).
    """
    if not news_items:
        trading_dates = price_df.index.normalize()
        return pd.Series(0.0, index=trading_dates, name="sentiment_score")

    # Score all items if not already scored
    if "sentiment_numeric" not in news_items[0]:
        news_items = score_news_items(news_items)

    records = []
    for item in news_items:
        pub_date = item.get("published_at") or item.get("created_at")
        if pub_date is None:
            continue
        if not hasattr(pub_date, "tzinfo") or pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=IST)

        trading_day = _get_trading_day(pub_date, price_df)
        numeric = item.get("sentiment_numeric", 0.0)

        # ── Base weight from match_type (company / sector / proxy) ────────
        match_type = item.get("match_type", "company")
        weight = MATCH_TYPE_WEIGHTS.get(match_type, 1.0)

        # ── Source-based multipliers ──────────────────────────────────────
        source = item.get("source", "").lower()

        # NSE official announcements get 2x on top of match_type weight
        if source in ("nse", "nse announcement", "announcement"):
            weight *= nse_announcement_weight

        # Reddit: weight by upvotes (log-scaled), multiplicative
        elif item.get("subreddit"):
            score = max(item.get("score", 1), 1)
            import math
            weight *= max(1.0, math.log10(score + 1))

        records.append(
            {
                "trading_day": trading_day.date(),
                "sentiment_numeric": numeric,
                "weight": weight,
                "match_type": match_type,
            }
        )

    if not records:
        trading_dates = price_df.index.normalize()
        return pd.Series(0.0, index=trading_dates, name="sentiment_score")

    df = pd.DataFrame(records)

    # Weighted average per trading day
    def weighted_avg(group: pd.DataFrame) -> float:
        total_weight = group["weight"].sum()
        if total_weight == 0:
            return 0.0
        return (group["sentiment_numeric"] * group["weight"]).sum() / total_weight

    daily_sentiment = df.groupby("trading_day").apply(weighted_avg)
    daily_sentiment.name = "sentiment_score"

    # Reindex to match price_df dates; forward-fill missing days
    price_dates = price_df.index.normalize().date
    full_index = pd.Index(price_dates)
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    daily_sentiment = daily_sentiment.reindex(
        pd.to_datetime(full_index), method=None
    )
    daily_sentiment = daily_sentiment.ffill().fillna(0.0)

    return daily_sentiment
