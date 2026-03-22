# NSE Simulator

A full-stack ML-powered NSE (National Stock Exchange, India) stock prediction simulator.
Add any NSE ticker and it fetches 2 years of historical data, scrapes Indian financial news,
scores sentiment with FinBERT, trains an XGBoost model, and shows interactive charts with
predicted price direction, buy/sell signals, and sentiment overlay.

---

## Architecture

```
Frontend (Next.js :3000)  ←→  Backend (FastAPI :8000)  ←→  SQLite DB
                                      ↓
                          yfinance | newspaper3k | PRAW | NewsAPI | NSE API
                                      ↓
                          Technical indicators (pandas-ta)
                          Sentiment scoring (FinBERT / HuggingFace)
                          XGBoost classifier (walk-forward CV)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- ~4 GB disk space (FinBERT model download on first run)

### 1. Clone / open the project

```bash
cd nse-simulator
```

### 2. Set up backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your API keys (see below)
```

### 3. Set up frontend

```bash
cd frontend
npm install
```

### 4. Start everything

From the **project root**:

```bash
chmod +x start.sh
./start.sh
```

Then open [http://localhost:3000](http://localhost:3000).

---

## API Keys Setup

### NewsAPI (required for macro Indian news)

1. Go to [https://newsapi.org/register](https://newsapi.org/register)
2. Sign up for a free account (100 requests/day on free tier)
3. Copy your API key into `backend/.env`:
   ```
   NEWSAPI_KEY=your_key_here
   ```

### Reddit API (required for r/IndiaInvestments and r/IndianStreetBets posts)

1. Log in to Reddit → go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **Create App** → choose **script** type
3. Set redirect URI to `http://localhost:8080`
4. Copy the client ID (under the app name) and secret:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=NSESimulator/1.0 by YourUsername
   ```

> **Note:** The app still works without these keys — news scraping from Economic Times and
> Moneycontrol will still function, just Reddit and NewsAPI sections will be skipped.

---

## First Run Notes

### FinBERT model download

On the first run, the backend will automatically download the `ProsusAI/finbert` model
from HuggingFace (~500 MB). This happens once and is cached locally at
`~/.cache/huggingface/transformers/`.

### Pipeline time

When you add a stock for the first time:
- Price data fetch: ~10 seconds
- News scraping (ET + Moneycontrol): ~1–2 minutes (polite delays)
- Reddit fetch: ~30 seconds
- FinBERT scoring: ~1–3 minutes (depends on hardware, GPU speeds this up significantly)
- XGBoost training: ~10–30 seconds
- **Total: 2–5 minutes**

The UI will show a "processing" state and auto-refresh every 15 seconds.

---

## Project Structure

```
nse-simulator/
├── backend/
│   ├── main.py                 # FastAPI app + all routes
│   ├── config.py               # Environment variables, constants
│   ├── data/
│   │   ├── price_fetcher.py    # yfinance OHLCV data (IST-aware)
│   │   ├── nse_announcements.py# NSE India public API
│   │   ├── news_scraper.py     # Economic Times + Moneycontrol scraper
│   │   ├── reddit_fetcher.py   # PRAW Reddit fetcher
│   │   └── macro_news.py       # NewsAPI macro Indian news
│   ├── features/
│   │   ├── technical.py        # RSI, MACD, Bollinger, EMA, ATR, OBV
│   │   ├── sentiment.py        # FinBERT scoring + daily aggregation
│   │   ├── calendar_flags.py   # RBI meetings, F&O expiry, budget day
│   │   └── feature_builder.py  # Merge all features, no-lookahead safe
│   ├── model/
│   │   ├── train.py            # XGBoost with walk-forward validation
│   │   ├── predict.py          # Generate historical predictions
│   │   ├── backtest.py         # Portfolio simulation vs buy-and-hold
│   │   └── signals.py          # Buy/sell/hold signal generation
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine + session
│   │   └── models.py           # SQLite tables
│   └── requirements.txt
├── frontend/
│   ├── pages/
│   │   ├── index.tsx           # Dashboard with stock grid
│   │   └── stock/[ticker].tsx  # Individual stock detail page
│   ├── components/
│   │   ├── StockChart.tsx      # Recharts ComposedChart with all overlays
│   │   ├── SentimentOverlay.tsx# FinBERT sentiment bar chart
│   │   ├── StockCard.tsx       # Dashboard card component
│   │   └── AddStockModal.tsx   # Add stock modal with validation
│   ├── lib/api.ts              # Typed API client (axios)
│   └── package.json
├── start.sh                    # One-command launcher
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stocks` | List all tracked stocks |
| POST | `/api/stocks` | Add a ticker (triggers pipeline) |
| DELETE | `/api/stocks/{ticker}` | Remove a stock |
| GET | `/api/stocks/{ticker}/chart` | Chart data (prices, predictions, signals) |
| GET | `/api/stocks/{ticker}/stats` | Backtest statistics |
| GET | `/api/stocks/{ticker}/news` | Recent news with sentiment |
| POST | `/api/stocks/{ticker}/refresh` | Re-run pipeline |
| GET | `/api/health` | Health check |

---

## Model Details

- **Algorithm:** XGBoost classifier (binary: up/down next day)
- **Validation:** Walk-forward (time-series safe, no random splits)
- **Features:** ~35 technical indicators + 5 calendar flags + 1 daily sentiment score
- **Signal threshold:** 60% confidence (configurable in `config.py`)
- **No lookahead bias:** all features on day T use only data available by EOD T

---

## Limitations & Disclaimer

- This is a **simulator for educational purposes only** — not financial advice
- Past model accuracy does not guarantee future returns
- NSE scraping may break if website layouts change
- Free NewsAPI tier has a 100 req/day limit
- FinBERT performance degrades on very short headlines
