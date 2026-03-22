from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB_URL, DB_PATH
from db.models import Base

_IS_SQLITE = DB_URL.startswith("sqlite")

# For SQLite, ensure the parent directory exists
if _IS_SQLITE and DB_PATH is not None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URL,
    # SQLite needs check_same_thread=False for FastAPI's async context;
    # PostgreSQL doesn't need (or accept) this arg.
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    # PostgreSQL: use a connection pool sized for the server
    pool_size=5 if not _IS_SQLITE else 5,
    max_overflow=10 if not _IS_SQLITE else 0,
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
    PostgreSQL supports  ALTER TABLE … ADD COLUMN IF NOT EXISTS.
    SQLite does not, so we catch the duplicate-column error silently there.
    """
    import sqlalchemy

    new_columns = [
        ("predictions", "predicted_price",      "FLOAT"),
        ("predictions", "predicted_price_low",  "FLOAT"),
        ("predictions", "predicted_price_high", "FLOAT"),
        ("predictions", "predicted_return_pct", "FLOAT"),
        ("predictions", "lstm_up_prob",         "FLOAT"),
    ]

    with engine.connect() as conn:
        for table, col, coltype in new_columns:
            if _IS_SQLITE:
                # SQLite: try/except because IF NOT EXISTS is not supported
                try:
                    conn.execute(sqlalchemy.text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"
                    ))
                    conn.commit()
                except Exception:
                    pass  # column already exists
            else:
                # PostgreSQL: native IF NOT EXISTS — no exception needed
                conn.execute(sqlalchemy.text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {coltype}"
                ))
                conn.commit()


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
