import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import IST
from data.macro_news import fetch_macro_news, fetch_ticker_news_via_newsapi
from data.news_scraper import fetch_all_news
from data.nse_announcements import fetch_corporate_announcements
from data.price_fetcher import fetch_price_data, get_company_name, get_latest_price
from data.reddit_fetcher import fetch_reddit_posts
from db.database import get_db, init_db
from db.models import NewsCache, PortfolioHolding, PortfolioSim, Prediction, Stock
from features.feature_builder import build_features
from features.sentiment import score_news_items
from model.backtest import run_backtest
from model.predict import generate_predictions, generate_future_predictions, predict_tomorrow
from model.signals import generate_signals
from model.train import model_exists, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("NSE Simulator backend started. Database initialised.")
    yield


app = FastAPI(
    title="NSE Simulator API",
    description="NSE stock prediction simulator with ML and sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class AddStockRequest(BaseModel):
    ticker: str


class AddHoldingRequest(BaseModel):
    ticker: str
    buy_price: float
    quantity: float
    notes: str | None = None


# ── Helper: full pipeline ──────────────────────────────────────────────────────


def _run_full_pipeline(ticker: str, _unused_db: Session = None) -> None:
    """
    Run the complete data fetch + feature engineering + model training pipeline.
    Called as a background task when a stock is first added or refreshed.
    Creates its own DB session to avoid using the request-scoped session
    that gets closed when the HTTP response is sent.
    """
    from db.database import SessionLocal
    db = SessionLocal()
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    logger.info("Starting full pipeline for %s", symbol)

    try:
        # ── 1. Fetch price data ───────────────────────────────────────────────
        price_df = fetch_price_data(symbol)

        # ── 2. Fetch news from all sources ────────────────────────────────────
        news_items: list[dict[str, Any]] = []
        try:
            news_items.extend(fetch_all_news(symbol))
        except Exception as exc:
            logger.warning("News scrape error: %s", exc)

        try:
            news_items.extend(fetch_reddit_posts(symbol))
        except Exception as exc:
            logger.warning("Reddit fetch error: %s", exc)

        try:
            news_items.extend(fetch_ticker_news_via_newsapi(symbol))
        except Exception as exc:
            logger.warning("NewsAPI fetch error: %s", exc)

        # ── 3. NSE announcements ──────────────────────────────────────────────
        announcements: list[dict[str, Any]] = []
        try:
            announcements = fetch_corporate_announcements(symbol)
        except Exception as exc:
            logger.warning("NSE announcements error: %s", exc)

        # ── 4. Score sentiment ────────────────────────────────────────────────
        if news_items:
            try:
                news_items = score_news_items(news_items)
            except Exception as exc:
                logger.warning("FinBERT scoring error: %s", exc)

        # ── 5. Persist news to DB ─────────────────────────────────────────────
        for item in news_items:
            try:
                pub = item.get("published_at") or item.get("created_at")
                if pub is None:
                    continue
                existing = (
                    db.query(NewsCache)
                    .filter(
                        NewsCache.ticker == symbol,
                        NewsCache.headline == item.get("headline", "")[:500],
                    )
                    .first()
                )
                if not existing:
                    db.add(
                        NewsCache(
                            ticker=symbol,
                            headline=(item.get("headline", "") or "")[:500],
                            body=(item.get("body", "") or item.get("text", "") or "")[:5000],
                            source=item.get("source", "unknown"),
                            published_at=pub if pub.tzinfo else pub.replace(tzinfo=IST),
                            sentiment_score=item.get("sentiment_numeric"),
                            sentiment_label=item.get("sentiment_label"),
                            url=item.get("url", ""),
                        )
                    )
            except Exception as exc:
                logger.debug("News DB insert error: %s", exc)
                db.rollback()
        db.commit()

        # ── 6. Build features ─────────────────────────────────────────────────
        feature_df = build_features(price_df, news_items, announcements, ticker=symbol)

        # ── 7. Train model ────────────────────────────────────────────────────
        model, metadata = train_model(symbol, feature_df)
        accuracy = metadata.get("accuracy", 0.0)

        # ── 8. Generate predictions ───────────────────────────────────────────
        pred_df = generate_predictions(symbol, feature_df)

        # ── 9. Add signals ────────────────────────────────────────────────────
        pred_df = generate_signals(pred_df)

        # ── 10. Persist predictions ───────────────────────────────────────────
        # Delete only 'updated' and 'future' rows — preserve 'initial' predictions
        # so the chart can always show what the model originally thought
        db.query(Prediction).filter(
            Prediction.ticker == symbol,
            Prediction.prediction_type.in_(["updated", "future"]),
        ).delete()

        # Check if this is the very first run (no 'initial' predictions yet)
        has_initial = (
            db.query(Prediction)
            .filter(Prediction.ticker == symbol, Prediction.prediction_type == "initial")
            .first()
        ) is not None

        for _, row in pred_df.iterrows():
            # Always write 'updated' predictions
            db.add(
                Prediction(
                    ticker=symbol,
                    date=row["date"],
                    actual_price=float(row["actual_price"]),
                    predicted_direction=int(row["predicted_direction"]),
                    confidence=float(row["prediction_confidence"]),
                    upper_band=float(row.get("upper_band", row["actual_price"])),
                    lower_band=float(row.get("lower_band", row["actual_price"])),
                    prediction_type="updated",
                )
            )
            # On first run, also store as 'initial' — this never gets overwritten
            if not has_initial:
                db.add(
                    Prediction(
                        ticker=symbol,
                        date=row["date"],
                        actual_price=float(row["actual_price"]),
                        predicted_direction=int(row["predicted_direction"]),
                        confidence=float(row["prediction_confidence"]),
                        upper_band=float(row.get("upper_band", row["actual_price"])),
                        lower_band=float(row.get("lower_band", row["actual_price"])),
                        prediction_type="initial",
                    )
                )

        # ── 10b. Generate tomorrow's ensemble price prediction ────────────────
        tomorrow: dict = {}
        try:
            tomorrow = predict_tomorrow(symbol, feature_df)
            logger.info(
                "Tomorrow prediction for %s: ₹%.2f (%.2f%%  conf=%.1f%%  model=%s)",
                symbol,
                tomorrow.get("predicted_price", 0),
                tomorrow.get("predicted_return_pct", 0),
                tomorrow.get("confidence", 0) * 100,
                tomorrow.get("model", "?"),
            )
        except Exception as exc:
            logger.warning("predict_tomorrow failed for %s: %s", symbol, exc)

        # ── 10c. Generate and persist future predictions (next 30 trading days) ──
        try:
            future_df = generate_future_predictions(symbol, feature_df, days=30)
            for i, (_, row) in enumerate(future_df.iterrows()):
                # Attach the precise ensemble prediction data to the first future row (tomorrow)
                is_tomorrow = (i == 0 and tomorrow)
                db.add(
                    Prediction(
                        ticker=symbol,
                        date=row["date"],
                        actual_price=None,
                        predicted_direction=int(row["predicted_direction"]),
                        confidence=float(row["prediction_confidence"]),
                        upper_band=float(row.get("upper_band", 0)),
                        lower_band=float(row.get("lower_band", 0)),
                        projected_price=float(row["projected_price"]),
                        prediction_type="future",
                        # Tomorrow-specific precise fields
                        # Tomorrow-specific precise fields
                        predicted_price=      (float(tomorrow["predicted_price"])      if tomorrow.get("predicted_price")      is not None else None) if is_tomorrow else None,
                        predicted_price_low=  (float(tomorrow["predicted_price_low"])  if tomorrow.get("predicted_price_low")  is not None else None) if is_tomorrow else None,
                        predicted_price_high= (float(tomorrow["predicted_price_high"]) if tomorrow.get("predicted_price_high") is not None else None) if is_tomorrow else None,
                        predicted_return_pct= (float(tomorrow["predicted_return_pct"]) if tomorrow.get("predicted_return_pct") is not None else None) if is_tomorrow else None,
                        lstm_up_prob=         (float(tomorrow["lstm_up_prob"])         if tomorrow.get("lstm_up_prob")         is not None else None) if is_tomorrow else None,
                    )
                )
            logger.info("Stored %d future predictions for %s", len(future_df), symbol)
        except Exception as exc:
            logger.warning("Future prediction generation failed for %s: %s", symbol, exc)

        db.commit()

        # ── 11. Run backtest on out-of-sample predictions only ────────────────
        # Using OOS fold predictions ensures we never evaluate on training data.
        # This gives honest Model Return / Sharpe / Alpha numbers.
        oos_df = metadata.get("oos_predictions")
        backtest_input = oos_df if (oos_df is not None and len(oos_df) > 30) else pred_df
        backtest = run_backtest(backtest_input)
        port_series = backtest["portfolio_series"]

        db.query(PortfolioSim).filter(PortfolioSim.ticker == symbol).delete()
        for _, row in port_series.iterrows():
            db.add(
                PortfolioSim(
                    ticker=symbol,
                    date=row["date"],
                    portfolio_value=float(row["portfolio_value"]),
                    benchmark_value=float(row["benchmark_value"]),
                )
            )
        db.commit()

        # ── 12. Update stock last_updated ─────────────────────────────────────
        stock = db.query(Stock).filter(Stock.ticker == symbol).first()
        if stock:
            stock.last_updated = datetime.now(tz=IST)
            db.commit()

        logger.info("Full pipeline completed for %s (accuracy=%.3f)", symbol, accuracy)

        # ── 13. Hyperparameter tuning (runs after pipeline, non-blocking) ─────
        # Optuna searches for better XGBoost params using walk-forward CV.
        # Results are saved to disk and loaded automatically on the next refresh.
        # The model improves incrementally with each Refresh Data click.
        try:
            from model.tune import tune_hyperparameters
            logger.info("Starting Optuna hyperparameter tuning for %s…", symbol)
            best = tune_hyperparameters(symbol, feature_df)
            if best:
                logger.info(
                    "Tuning done for %s — best params: lr=%.4f, depth=%s, n_est=%s",
                    symbol,
                    best.get("learning_rate", 0),
                    best.get("max_depth", "?"),
                    best.get("n_estimators", "?"),
                )
        except Exception as exc:
            logger.warning("Hyperparameter tuning failed for %s (non-fatal): %s", symbol, exc)

    except Exception as exc:
        logger.exception("Pipeline failed for %s: %s", symbol, exc)
        db.rollback()
    finally:
        db.close()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/api/stocks")
def list_stocks(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List all tracked stocks."""
    from model.train import load_model, model_exists
    stocks = db.query(Stock).order_by(Stock.added_at.desc()).all()
    results = []
    for s in stocks:
        # Latest historical prediction (for current price display)
        last_pred = (
            db.query(Prediction)
            .filter(
                Prediction.ticker == s.ticker,
                Prediction.prediction_type != "future",
            )
            .order_by(Prediction.date.desc())
            .first()
        )

        # Use walk-forward CV accuracy from saved model metadata — this is the
        # honest out-of-sample accuracy, NOT the in-sample backtest win rate.
        accuracy = 0.5
        if model_exists(s.ticker):
            try:
                _, meta = load_model(s.ticker)
                accuracy = meta.get("accuracy", 0.5)
            except Exception:
                pass

        results.append(
            {
                "ticker": s.ticker,
                "display_name": s.display_name,
                "added_at": s.added_at.isoformat() if s.added_at else None,
                "last_updated": s.last_updated.isoformat() if s.last_updated else None,
                "last_price": last_pred.actual_price if last_pred else None,
                "model_accuracy": round(accuracy * 100, 1),
            }
        )
    return results


def _compute_accuracy_from_db(predictions: list) -> float:
    """Compute directional accuracy from consecutive prediction records.

    Only uses historical predictions with real actual_price values.
    Skips future predictions (actual_price is None by design).
    """
    if len(predictions) < 2:
        return 0.5

    # Filter to only historical rows that have real prices
    historical = [
        p for p in predictions
        if p.actual_price is not None
        and getattr(p, "prediction_type", "updated") != "future"
    ]
    if len(historical) < 2:
        return 0.5

    sorted_preds = sorted(historical, key=lambda p: p.date)
    correct = 0
    total = 0
    for i in range(len(sorted_preds) - 1):
        today = sorted_preds[i]
        tomorrow = sorted_preds[i + 1]
        if today.predicted_direction is None:
            continue
        actual_direction = 1 if tomorrow.actual_price > today.actual_price else 0
        if today.predicted_direction == actual_direction:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.5


@app.post("/api/stocks", status_code=202)
async def add_stock(
    request: AddStockRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Add a new ticker and trigger full data fetch + model training."""
    ticker = request.ticker.upper().replace(".NS", "").replace(".BO", "")

    # Check if already exists
    existing = db.query(Stock).filter(Stock.ticker == ticker).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"{ticker} is already being tracked")

    # Validate ticker by attempting to get company name
    try:
        company_name = get_company_name(ticker)
    except Exception:
        company_name = ticker

    # Create stock record immediately
    stock = Stock(
        ticker=ticker,
        display_name=company_name,
        added_at=datetime.now(tz=IST),
    )
    db.add(stock)
    db.commit()

    # Run pipeline in background (creates its own DB session)
    background_tasks.add_task(_run_full_pipeline, ticker)

    return {
        "ticker": ticker,
        "display_name": company_name,
        "status": "processing",
        "message": "Data fetch and model training started. This may take 2-3 minutes.",
    }


@app.delete("/api/stocks/{ticker}", status_code=200)
def remove_stock(ticker: str, db: Session = Depends(get_db)) -> dict[str, str]:
    """Remove a stock from tracking."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    db.query(Prediction).filter(Prediction.ticker == symbol).delete()
    db.query(NewsCache).filter(NewsCache.ticker == symbol).delete()
    db.query(PortfolioSim).filter(PortfolioSim.ticker == symbol).delete()
    db.delete(stock)
    db.commit()

    return {"message": f"{symbol} removed from tracking"}


@app.get("/api/stocks/{ticker}/chart")
def get_chart_data(ticker: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return all chart data: prices, predictions, signals, sentiment, portfolio."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    # Fetch all 3 prediction types separately
    updated_preds = (
        db.query(Prediction)
        .filter(Prediction.ticker == symbol, Prediction.prediction_type == "updated")
        .order_by(Prediction.date.asc())
        .all()
    )
    initial_preds = (
        db.query(Prediction)
        .filter(Prediction.ticker == symbol, Prediction.prediction_type == "initial")
        .order_by(Prediction.date.asc())
        .all()
    )
    future_preds = (
        db.query(Prediction)
        .filter(Prediction.ticker == symbol, Prediction.prediction_type == "future")
        .order_by(Prediction.date.asc())
        .all()
    )

    predictions = updated_preds  # keep backward compat reference
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions available for {symbol}. Pipeline may still be running.")

    news_with_sentiment = (
        db.query(NewsCache)
        .filter(NewsCache.ticker == symbol, NewsCache.sentiment_score.isnot(None))
        .order_by(NewsCache.published_at.asc())
        .all()
    )

    portfolio = (
        db.query(PortfolioSim)
        .filter(PortfolioSim.ticker == symbol)
        .order_by(PortfolioSim.date.asc())
        .all()
    )

    # Build sentiment series (daily aggregated)
    sentiment_by_date: dict[str, float] = {}
    for n in news_with_sentiment:
        date_key = n.published_at.strftime("%Y-%m-%d") if n.published_at else None
        if date_key and n.sentiment_score is not None:
            if date_key not in sentiment_by_date:
                sentiment_by_date[date_key] = []
            sentiment_by_date[date_key].append(n.sentiment_score)

    sentiment_daily = {k: sum(v) / len(v) for k, v in sentiment_by_date.items()}

    # Serialise predictions + signals
    chart_data = []
    for p in predictions:
        date_str = p.date.strftime("%Y-%m-%d") if p.date else None
        signal = "hold"
        if p.predicted_direction is not None and p.confidence is not None:
            from config import SIGNAL_CONFIDENCE_THRESHOLD
            if p.confidence > SIGNAL_CONFIDENCE_THRESHOLD:
                signal = "buy" if p.predicted_direction == 1 else "sell"

        chart_data.append(
            {
                "date": date_str,
                "actual_price": p.actual_price,
                "predicted_direction": p.predicted_direction,
                "confidence": p.confidence,
                "upper_band": p.upper_band,
                "lower_band": p.lower_band,
                "signal": signal,
                "sentiment_score": sentiment_daily.get(date_str, 0.0),
            }
        )

    portfolio_data = [
        {
            "date": pt.date.strftime("%Y-%m-%d"),
            "portfolio_value": pt.portfolio_value,
            "benchmark_value": pt.benchmark_value,
        }
        for pt in portfolio
    ]

    # Build initial predictions lookup by date
    initial_by_date = {
        p.date.strftime("%Y-%m-%d"): {
            "predicted_direction": p.predicted_direction,
            "confidence": p.confidence,
        }
        for p in initial_preds
    }

    # Future predictions series (next 30 days)
    tomorrow_prediction: dict | None = None
    future_data = []
    for i, p in enumerate(future_preds):
        date_str = p.date.strftime("%Y-%m-%d") if p.date else None
        row = {
            "date": date_str,
            "projected_price": p.projected_price,
            "predicted_direction": p.predicted_direction,
            "confidence": p.confidence,
            "upper_band": p.upper_band,
            "lower_band": p.lower_band,
        }
        future_data.append(row)

        # First future row = tomorrow — extract precise ensemble prediction
        if i == 0 and p.predicted_price is not None:
            tomorrow_prediction = {
                "date":                date_str,
                "predicted_price":     p.predicted_price,
                "predicted_price_low": p.predicted_price_low,
                "predicted_price_high":p.predicted_price_high,
                "predicted_return_pct":p.predicted_return_pct,
                "direction":           p.predicted_direction,
                "confidence":          p.confidence,
                "lstm_up_prob":        p.lstm_up_prob,
            }

    # Add initial prediction fields to chart_data
    for entry in chart_data:
        initial = initial_by_date.get(entry["date"], {})
        entry["initial_predicted_direction"] = initial.get("predicted_direction")
        entry["initial_confidence"] = initial.get("confidence")

    return {
        "ticker": symbol,
        "display_name": stock.display_name,
        "chart_data": chart_data,              # historical: actual + updated predictions
        "future_data": future_data,            # next 30 days: projected prices
        "portfolio_data": portfolio_data,
        "tomorrow_prediction": tomorrow_prediction,  # precise next-day price estimate
    }


@app.get("/api/stocks/{ticker}/stats")
def get_stats(ticker: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return backtest statistics for a ticker."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    portfolio = (
        db.query(PortfolioSim)
        .filter(PortfolioSim.ticker == symbol)
        .order_by(PortfolioSim.date.asc())
        .all()
    )

    if not portfolio:
        raise HTTPException(status_code=404, detail="No backtest data available")

    port_values = [p.portfolio_value for p in portfolio]
    bench_values = [p.benchmark_value for p in portfolio]

    from model.backtest import _max_drawdown, _sharpe_ratio
    import pandas as pd

    port_series = pd.Series(port_values)
    bench_series = pd.Series(bench_values)
    port_returns = port_series.pct_change().dropna()
    bench_returns = bench_series.pct_change().dropna()

    # Accuracy: walk-forward CV accuracy from model metadata (honest out-of-sample)
    accuracy = 0.5
    try:
        from model.train import load_model as _load_model
        _, meta = _load_model(symbol)
        accuracy = meta.get("accuracy", 0.5)
    except Exception:
        pass

    cumulative_return = (port_values[-1] - 100_000) / 100_000 * 100
    benchmark_return = (bench_values[-1] - 100_000) / 100_000 * 100

    return {
        "ticker": symbol,
        "cumulative_return": round(cumulative_return, 2),
        "benchmark_return": round(benchmark_return, 2),
        "alpha": round(cumulative_return - benchmark_return, 2),
        "sharpe_ratio": round(_sharpe_ratio(port_returns), 3),
        "max_drawdown": round(_max_drawdown(port_series) * 100, 2),
        "model_accuracy": round(accuracy * 100, 1),
        "total_trading_days": len(portfolio),
    }


@app.get("/api/stocks/{ticker}/news")
def get_news(
    ticker: str,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return recent news with sentiment scores for a ticker."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    news = (
        db.query(NewsCache)
        .filter(NewsCache.ticker == symbol)
        .order_by(NewsCache.published_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "ticker": symbol,
        "news": [
            {
                "headline": n.headline,
                "source": n.source,
                "published_at": n.published_at.isoformat() if n.published_at else None,
                "sentiment_score": n.sentiment_score,
                "sentiment_label": n.sentiment_label,
                "url": n.url,
            }
            for n in news
        ],
    }


@app.post("/api/stocks/{ticker}/refresh", status_code=202)
async def refresh_stock(
    ticker: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Re-fetch data and re-run the model for a ticker."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    background_tasks.add_task(_run_full_pipeline, symbol)
    return {"message": f"Refresh started for {symbol}. This may take 2-3 minutes."}


@app.get("/api/stocks/{ticker}/live")
def get_live_price(ticker: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Return the current live market price for a ticker plus intraday stats.
    Also computes how today's actual movement compares to the model's prediction.

    Designed to be polled every 5–30 seconds from the frontend Live Mode toggle.
    Yahoo Finance data refreshes ~every 15 seconds during market hours.
    """
    from data.live_price_fetcher import fetch_live_price

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    stock = db.query(Stock).filter(Stock.ticker == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    live = fetch_live_price(symbol)

    # Pull the model's prediction for today (first future prediction = tomorrow,
    # which IS today if we're inside trading hours) from the DB
    tomorrow_pred = (
        db.query(Prediction)
        .filter(
            Prediction.ticker == symbol,
            Prediction.prediction_type == "future",
            Prediction.predicted_price.isnot(None),
        )
        .order_by(Prediction.date.asc())
        .first()
    )

    model_comparison: dict | None = None
    if tomorrow_pred and live.get("price") and live.get("prev_close"):
        actual_direction   = 1 if live["price"] > live["prev_close"] else 0
        predicted_direction = tomorrow_pred.predicted_direction
        is_correct         = (actual_direction == predicted_direction)
        model_comparison   = {
            "predicted_direction":  predicted_direction,        # 1=up 0=down
            "predicted_price":      tomorrow_pred.predicted_price,
            "predicted_price_low":  tomorrow_pred.predicted_price_low,
            "predicted_price_high": tomorrow_pred.predicted_price_high,
            "predicted_return_pct": tomorrow_pred.predicted_return_pct,
            "confidence":           tomorrow_pred.confidence,
            "actual_direction_now": actual_direction,
            "prediction_correct":   is_correct,
            "gap_to_target":        round(
                (tomorrow_pred.predicted_price or 0) - (live["price"] or 0), 2
            ),
        }

    return {**live, "model_comparison": model_comparison}


# ── Intraday (10-min) Training & Prediction Routes ───────────────────────────


class RecordActualRequest(BaseModel):
    actual_price: float


@app.get("/api/stocks/{ticker}/intraday")
def get_intraday(ticker: str) -> dict[str, Any]:
    """
    Return today's 10-min OHLCV bars, the current intraday prediction,
    and session accuracy stats. Designed to be polled every 30s by the
    frontend 'Today' chart tab.
    """
    from data.intraday_fetcher import fetch_today_bars
    from model.intraday_trainer import predict_next_bar, get_session_accuracy

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    bars = fetch_today_bars(symbol)
    session_acc = get_session_accuracy(symbol)

    bars_data: list[dict] = []
    prediction: dict | None = None

    if not bars.empty:
        for ts, row in bars.iterrows():
            bars_data.append({
                "timestamp": ts.isoformat(),
                "open":   round(float(row["open"]),   2),
                "high":   round(float(row["high"]),   2),
                "low":    round(float(row["low"]),    2),
                "close":  round(float(row["close"]),  2),
                "volume": int(row["volume"]),
            })
        pred = predict_next_bar(symbol, bars)
        if pred and "error" not in pred:
            prediction = pred

    return {
        "ticker":           symbol,
        "bars":             bars_data,
        "prediction":       prediction,
        "session_accuracy": session_acc,
    }


@app.post("/api/stocks/{ticker}/intraday/train")
def train_intraday_endpoint(ticker: str) -> dict[str, Any]:
    """Fetch last 5 days of 10-min bars and (re)train the lightweight intraday model."""
    from data.intraday_fetcher import fetch_intraday_bars
    from model.intraday_trainer import train_intraday as _train_intraday

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    bars = fetch_intraday_bars(symbol, days_back=5)
    if bars.empty:
        raise HTTPException(status_code=400, detail="No intraday data available")

    result = _train_intraday(symbol, bars)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"ticker": symbol, **result}


@app.post("/api/stocks/{ticker}/intraday/predict")
def predict_intraday_endpoint(ticker: str) -> dict[str, Any]:
    """
    Make a next-10-min prediction using the intraday model and record it
    in today's session log. Call this after /intraday/train.
    """
    from data.intraday_fetcher import fetch_today_bars
    from model.intraday_trainer import predict_next_bar, record_prediction

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    bars = fetch_today_bars(symbol)
    if bars.empty:
        raise HTTPException(status_code=400, detail="No intraday bars available for today")

    pred = predict_next_bar(symbol, bars)
    if "error" in pred:
        raise HTTPException(status_code=400, detail=pred["error"])

    record_prediction(symbol, pred)
    return {"ticker": symbol, **pred}


@app.post("/api/stocks/{ticker}/intraday/record-actual")
def record_intraday_actual(
    ticker: str, request: RecordActualRequest,
) -> dict[str, Any]:
    """
    Record the actual close price for the last pending 10-min prediction.
    Returns whether the prediction was correct and current session accuracy.
    """
    from model.intraday_trainer import record_actual, get_session_accuracy

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    correct = record_actual(symbol, request.actual_price)
    session = get_session_accuracy(symbol)
    return {
        "ticker":            symbol,
        "prediction_correct": correct,
        "session_accuracy":  session,
    }


@app.post("/api/stocks/{ticker}/intraday/replay")
def replay_intraday_endpoint(ticker: str, days_back: int = 5) -> dict[str, Any]:
    """
    Offline replay mode (for when market is closed).
    Walks through the last N days of 10-min bars sequentially:
      train on bars 0..i → predict bar i+1 → compare actual → advance.
    Returns cumulative replay accuracy.
    """
    import pandas as pd
    from data.intraday_fetcher import fetch_intraday_bars, split_intraday_for_replay
    from model.intraday_trainer import (
        train_intraday as _train_intraday,
        predict_next_bar,
    )

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    all_bars = fetch_intraday_bars(symbol, days_back=days_back)
    if all_bars.empty:
        raise HTTPException(status_code=400, detail="No intraday data available")

    day_groups = split_intraday_for_replay(all_bars)
    total_preds   = 0
    correct_preds = 0
    cumulative: pd.DataFrame | None = None

    for day_df in day_groups:
        for i in range(15, len(day_df) - 1):
            train_bars = (
                pd.concat([cumulative, day_df.iloc[:i]]) if cumulative is not None
                else day_df.iloc[:i]
            )
            result = _train_intraday(symbol, train_bars)
            if "error" in result:
                continue
            pred = predict_next_bar(symbol, train_bars)
            if "error" in pred:
                continue
            # Compare prediction against the actual next bar
            actual_close = float(day_df.iloc[i + 1]["close"])
            prev_close   = float(day_df.iloc[i]["close"])
            actual_dir   = 1 if actual_close > prev_close else 0
            total_preds += 1
            if actual_dir == pred["direction"]:
                correct_preds += 1

        cumulative = (
            pd.concat([cumulative, day_df]) if cumulative is not None else day_df
        )

    accuracy = round(correct_preds / total_preds, 4) if total_preds > 0 else None
    return {
        "ticker":              symbol,
        "days_replayed":       len(day_groups),
        "total_predictions":   total_preds,
        "correct_predictions": correct_preds,
        "replay_accuracy":     accuracy,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "NSE Simulator API"}


# ── Portfolio Holdings Routes ─────────────────────────────────────────────────


@app.get("/api/portfolio")
def list_holdings(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List all portfolio holdings with current P&L."""
    holdings = db.query(PortfolioHolding).order_by(PortfolioHolding.added_at.desc()).all()
    results = []

    for h in holdings:
        # Get latest actual price from predictions
        latest_pred = (
            db.query(Prediction)
            .filter(Prediction.ticker == h.ticker, Prediction.prediction_type == "updated")
            .order_by(Prediction.date.desc())
            .first()
        )
        # Get current model signal
        current_signal = "hold"
        current_confidence = None
        if latest_pred:
            from config import SIGNAL_CONFIDENCE_THRESHOLD
            if latest_pred.confidence and latest_pred.confidence > SIGNAL_CONFIDENCE_THRESHOLD:
                current_signal = "buy" if latest_pred.predicted_direction == 1 else "sell"
            current_confidence = latest_pred.confidence

        current_price = latest_pred.actual_price if latest_pred else None
        cost_basis = h.buy_price * h.quantity
        current_value = (current_price * h.quantity) if current_price else None
        pnl_abs = (current_value - cost_basis) if current_value else None
        pnl_pct = ((current_price - h.buy_price) / h.buy_price * 100) if current_price else None

        results.append({
            "ticker": h.ticker,
            "buy_price": h.buy_price,
            "quantity": h.quantity,
            "notes": h.notes,
            "added_at": h.added_at.isoformat() if h.added_at else None,
            "current_price": current_price,
            "current_value": round(current_value, 2) if current_value else None,
            "cost_basis": round(cost_basis, 2),
            "pnl_abs": round(pnl_abs, 2) if pnl_abs is not None else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "current_signal": current_signal,
            "signal_confidence": round(current_confidence * 100, 1) if current_confidence else None,
        })

    return results


@app.post("/api/portfolio", status_code=201)
def add_holding(
    request: AddHoldingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Add a holding to the portfolio. Auto-adds the stock to the simulator if not already tracked."""
    ticker = request.ticker.upper().replace(".NS", "").replace(".BO", "")

    # Check if holding already exists
    existing = db.query(PortfolioHolding).filter(PortfolioHolding.ticker == ticker).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"{ticker} is already in your portfolio")

    # Add holding
    holding = PortfolioHolding(
        ticker=ticker,
        buy_price=request.buy_price,
        quantity=request.quantity,
        notes=request.notes,
        added_at=datetime.now(tz=IST),
    )
    db.add(holding)
    db.commit()

    # Auto-add to simulator if not already tracked
    stock_exists = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock_exists:
        try:
            company_name = get_company_name(ticker)
        except Exception:
            company_name = ticker
        stock = Stock(
            ticker=ticker,
            display_name=company_name,
            added_at=datetime.now(tz=IST),
        )
        db.add(stock)
        db.commit()
        background_tasks.add_task(_run_full_pipeline, ticker)

    return {
        "ticker": ticker,
        "buy_price": request.buy_price,
        "quantity": request.quantity,
        "message": "Holding added. Model training started if stock was not already tracked.",
    }


@app.put("/api/portfolio/{ticker}")
def update_holding(
    ticker: str,
    request: AddHoldingRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update buy price or quantity for an existing holding."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    holding = db.query(PortfolioHolding).filter(PortfolioHolding.ticker == symbol).first()
    if not holding:
        raise HTTPException(status_code=404, detail=f"{symbol} not in portfolio")

    holding.buy_price = request.buy_price
    holding.quantity = request.quantity
    if request.notes is not None:
        holding.notes = request.notes
    db.commit()

    return {"ticker": symbol, "buy_price": holding.buy_price, "quantity": holding.quantity}


@app.delete("/api/portfolio/{ticker}")
def remove_holding(ticker: str, db: Session = Depends(get_db)) -> dict[str, str]:
    """Remove a holding from the portfolio (does not remove from simulator)."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    holding = db.query(PortfolioHolding).filter(PortfolioHolding.ticker == symbol).first()
    if not holding:
        raise HTTPException(status_code=404, detail=f"{symbol} not in portfolio")
    db.delete(holding)
    db.commit()
    return {"message": f"{symbol} removed from portfolio"}
