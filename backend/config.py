import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from backend directory
load_dotenv(Path(__file__).parent / ".env")

# Timezone constant used across the entire app
IST = ZoneInfo("Asia/Kolkata")

# Database — local SQLite by default, cloud PostgreSQL if DATABASE_URL is set.
# To use a cloud database (e.g. Supabase), set DATABASE_URL in .env:
#   DATABASE_URL=postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
_DATABASE_URL_ENV = ""  # Forcibly fallback to SQLite to bypass network firewalls
if _DATABASE_URL_ENV:
    # Cloud PostgreSQL (Supabase, Neon, Railway, etc.)
    # Supabase connection strings start with "postgres://" — SQLAlchemy needs "postgresql://"
    DB_URL = _DATABASE_URL_ENV.replace("postgres://", "postgresql://", 1)
    DB_PATH = None   # not used for PostgreSQL
else:
    # Local SQLite fallback
    DB_PATH = Path(__file__).parent / "data" / "nse_simulator.db"
    DB_URL = f"sqlite:///{DB_PATH}"

# API Keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "NSESimulator/1.0")

# HuggingFace model
FINBERT_MODEL = "ProsusAI/finbert"

# NSE API base
NSE_API_BASE = "https://www.nseindia.com/api"

# Model storage
MODEL_DIR = Path(__file__).parent / "model" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Scraping
SCRAPE_DELAY_MIN = 1.0
SCRAPE_DELAY_MAX = 2.0

# NSE market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Model params
WALK_FORWARD_WINDOW = 252  # trading days (~1 year)
SIGNAL_CONFIDENCE_THRESHOLD = 0.60
HISTORY_YEARS = 5
