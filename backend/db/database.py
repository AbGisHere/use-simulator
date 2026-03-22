from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB_URL
from db.models import Base

# Ensure parent directory exists
Path(DB_URL.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they don't exist, and migrate new columns."""
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def _run_migrations() -> None:
    """
    Add new columns to existing tables without data loss.
    SQLite does not support ALTER TABLE ADD COLUMN IF NOT EXISTS,
    so we catch the 'duplicate column' error silently.
    """
    new_columns = [
        ("predictions", "predicted_price",      "FLOAT"),
        ("predictions", "predicted_price_low",  "FLOAT"),
        ("predictions", "predicted_price_high", "FLOAT"),
        ("predictions", "predicted_return_pct", "FLOAT"),
        ("predictions", "lstm_up_prob",          "FLOAT"),
        # target_return is only a feature-df column, not stored in DB — no migration needed
    ]
    with engine.connect() as conn:
        for table, col, coltype in new_columns:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # Column already exists


@contextmanager
def get_session():
    """Context manager for database sessions with automatic cleanup."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
