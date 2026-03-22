from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    actual_price = Column(Float, nullable=True)       # null for future dates
    predicted_direction = Column(Integer, nullable=True)  # 1=up, 0=down
    confidence = Column(Float, nullable=True)
    upper_band = Column(Float, nullable=True)
    lower_band = Column(Float, nullable=True)

    # Which "version" of prediction this is:
    #   'initial'  — stored on first model run, never overwritten (shows original forecast)
    #   'updated'  — overwritten on each Refresh (shows current model's view of history)
    #   'future'   — projected prices beyond today (next 30 trading days)
    prediction_type = Column(String(20), nullable=False, default="updated")

    # For future predictions: the projected price (actual_price will be null)
    projected_price = Column(Float, nullable=True)

    # Tomorrow's ensemble price prediction (set on the nearest future prediction only)
    predicted_price      = Column(Float, nullable=True)   # point estimate (₹)
    predicted_price_low  = Column(Float, nullable=True)   # lower band (₹)
    predicted_price_high = Column(Float, nullable=True)   # upper band (₹)
    predicted_return_pct = Column(Float, nullable=True)   # expected % move
    lstm_up_prob         = Column(Float, nullable=True)   # LSTM P(up) for transparency

    __table_args__ = (
        UniqueConstraint("ticker", "date", "prediction_type", name="uq_prediction_ticker_date_type"),
    )


class NewsCache(Base):
    __tablename__ = "news_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    body = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    url = Column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("ticker", "headline", "published_at", name="uq_news_item"),)


class PortfolioHolding(Base):
    """A real holding entered by the user — their actual buy price and quantity."""
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    buy_price = Column(Float, nullable=False)       # price paid per share (₹)
    quantity = Column(Float, nullable=False)        # number of shares
    added_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class PortfolioSim(Base):
    __tablename__ = "portfolio_sim"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    portfolio_value = Column(Float, nullable=False)
    benchmark_value = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_portfolio_ticker_date"),)
