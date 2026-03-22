import logging
from datetime import datetime
from typing import Any

import praw

from config import IST, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)

SUBREDDITS = ["IndiaInvestments", "IndianStreetBets"]
MAX_POSTS_PER_SUB = 25
MAX_COMMENTS_PER_POST = 5


def _utc_to_ist(utc_timestamp: float) -> datetime:
    """Convert a UTC Unix timestamp to an IST-aware datetime."""
    from zoneinfo import ZoneInfo

    dt_utc = datetime.utcfromtimestamp(utc_timestamp).replace(tzinfo=ZoneInfo("UTC"))
    return dt_utc.astimezone(IST)


def _build_reddit_client() -> praw.Reddit:
    """Build a read-only PRAW Reddit client."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise RuntimeError(
            "Reddit API credentials not configured. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
        )
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        read_only=True,
    )


def fetch_reddit_posts(ticker: str, max_age_days: int = 90) -> list[dict[str, Any]]:
    """
    Fetch Reddit posts mentioning a ticker from r/IndiaInvestments and r/IndianStreetBets.

    Returns list of dicts:
        {text, score, created_at (IST datetime), subreddit, url}
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    results: list[dict[str, Any]] = []

    try:
        reddit = _build_reddit_client()
    except RuntimeError as exc:
        logger.warning("Reddit client unavailable: %s", exc)
        return results

    for subreddit_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Search by both symbol and common name variants
            search_queries = [symbol, f"${symbol}", symbol.lower()]

            seen_ids: set[str] = set()
            for query in search_queries:
                try:
                    for submission in subreddit.search(
                        query, sort="new", time_filter="year", limit=MAX_POSTS_PER_SUB
                    ):
                        if submission.id in seen_ids:
                            continue
                        seen_ids.add(submission.id)

                        created_ist = _utc_to_ist(submission.created_utc)

                        # Build combined text: title + top comments
                        text_parts = [submission.title]
                        if submission.selftext:
                            text_parts.append(submission.selftext[:1000])

                        # Top comments
                        submission.comments.replace_more(limit=0)
                        for comment in list(submission.comments)[:MAX_COMMENTS_PER_POST]:
                            if hasattr(comment, "body"):
                                text_parts.append(comment.body[:500])

                        combined_text = "\n".join(text_parts)

                        results.append(
                            {
                                "ticker": symbol,
                                "text": combined_text,
                                "score": submission.score,
                                "created_at": created_ist,
                                "subreddit": subreddit_name,
                                "url": f"https://reddit.com{submission.permalink}",
                                "headline": submission.title,
                            }
                        )
                except Exception as exc:
                    logger.debug("Reddit search error for query %s in %s: %s", query, subreddit_name, exc)
                    continue

        except Exception as exc:
            logger.warning("Reddit error for subreddit %s: %s", subreddit_name, exc)
            continue

    # Sort by date descending
    results.sort(key=lambda x: x["created_at"], reverse=True)
    logger.info("Fetched %d Reddit posts for %s", len(results), symbol)
    return results
