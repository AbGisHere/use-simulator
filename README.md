# NSE Stock Simulator

A full-stack, self-improving ML trading simulator for NSE (National Stock Exchange of India) stocks. Add any NSE ticker and the system fetches years of historical data, scrapes news, scores sentiment with FinBERT, trains an XGBoost + LSTM ensemble, and continuously improves itself through Optuna hyperparameter search — all while tracking predictions against real prices in real time.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [How It Works — End to End](#how-it-works--end-to-end)
3. [The Auto-Bot](#the-auto-bot)
4. [The Models](#the-models)
5. [Features Explained](#features-explained)
6. [Requirements & Setup](#requirements--setup)
7. [Running Everything](#running-everything)
8. [The Dashboard](#the-dashboard)
9. [Cloud Database Sync](#cloud-database-sync)
10. [Architecture](#architecture)
11. [API Reference](#api-reference)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)

---

## What It Does

You type in a ticker like `TCS` or `PVRINOX`. The system:

- Fetches **5+ years** of daily price data and builds a full feature set (technical indicators, institutional flows, sentiment, sector signals).
- Trains an **XGBoost + LSTM ensemble** and runs **Optuna hyperparameter search** to find the best model parameters for that specific stock.
- Shows you a **price chart** with the model's historical predictions overlaid, a **30-day forward forecast**, and **buy/sell signals**.
- Runs a **10-minute intraday prediction loop** during market hours — predicting where the stock goes every 10 minutes, checking whether it was right, and retraining on the latest data.
- Tracks a **live price feed** you can toggle on — showing the model's prediction vs what the stock is actually doing right now.
- Keeps all data in a **cloud PostgreSQL database** (Supabase) so everything syncs between your Mac and Windows machines.

---

## How It Works — End to End

### When you add a stock

1. **Price data** — 5 years of daily OHLCV is downloaded from Yahoo Finance.
2. **News** — headlines are scraped from Indian financial news sites, NewsAPI, Reddit (r/IndiaInvestments), and NSE corporate announcements.
3. **Sentiment** — every headline is scored by **FinBERT** (a financial BERT model) into `bullish / bearish / neutral` with a numeric score. These scores become model features.
4. **Institutional data** — FII/DII (Foreign/Domestic Institutional Investor) net buy/sell flow is fetched from NSE's API. This tells you whether institutions are accumulating or distributing.
5. **Delivery %** — fetched from NSE's daily bhav copy archive. High delivery % means conviction buying/selling rather than intraday noise.
6. **Sector signals** — the Nifty sector index that the stock belongs to (e.g. `^CNXIT` for IT stocks) is fetched. Its 5-day and 20-day returns and percentile rank become features — this captures sector rotation.
7. **Feature engineering** — all of the above is combined into ~60 features per trading day.
8. **Walk-forward training** — the model is trained using expanding-window cross-validation (no lookahead bias). Each fold trains on everything up to a point, validates on the next 63 trading days, and the accuracy score reported is the average across all out-of-sample folds.
9. **Optuna tuning** — 35 Optuna trials run in the background, searching across 16 XGBoost hyperparameters to find the best combination for this specific stock. Studies are saved to disk and every subsequent refresh adds more trials — the model keeps getting smarter.
10. **LSTM training** — a bidirectional 2-layer LSTM is trained on the same feature set to capture sequential patterns XGBoost misses.
11. **Predictions** — the ensemble (55% XGBoost + 45% LSTM) generates historical predictions (for the chart), a 30-day forward forecast, and a precise price estimate for tomorrow.
12. **Backtest** — a simulated portfolio starting at ₹1,00,000 trades the model's signals out-of-sample. The reported Sharpe ratio, alpha, drawdown, and model return all come from this honest OOS backtest.

### On every refresh

The entire pipeline above re-runs. Optuna adds more trials to the existing study — it's not starting from scratch, it's building on everything it learned before. Over time, the model converges on the optimal hyperparameters for each stock.

---

## The Auto-Bot

`auto_trainer.py` is a background process that runs indefinitely and drives the entire self-improvement loop. You start it once and leave it running.

### During market hours (9:15 AM – 3:30 PM IST, Mon–Fri)

The bot enters a **10-minute intraday prediction loop**:

```
Initial setup:
  └── For each stock: fetch 5 days of 10-min bars → train intraday model

Every 10 minutes:
  ├── For each stock: predict the next bar's direction + price
  ├── Display prediction table (ticker | direction | confidence | predicted price | range)
  ├── Wait for the next 10-min bar to close (~10 min + buffer)
  ├── Fetch actual live prices
  ├── Record whether each prediction was correct
  ├── Retrain on the new bar
  └── Display session accuracy table (ticker | correct/wrong | session %)
```

The intraday model trains in under 500ms per stock, so retraining after every bar is fast. The session accuracy resets every day at midnight.

### After market close

```
1. Offline replay
   └── For each stock, walk through last 5 days of 10-min bars:
       train on history → predict next bar → check actual → retrain → advance
       (measures what intraday accuracy would have been)

2. Overnight Optuna retrain
   └── For each stock: run full pipeline + 35–60 new Optuna trials
       (builds on previous study knowledge, converges toward optimal params)

3. Random discovery (every 2 cycles)
   └── Pick a random stock from 60+ NSE universe tickers and add it
       (the bot explores stocks you haven't manually added)

4. Sleep until 9:10 AM IST on next trading day
```

### What "self-improving" means in practice

- **Cycle 1** (day 1): Optuna has 35 trials. It's still exploring broadly.
- **Cycle 10** (day 5): 350 trials accumulated. TPE sampler has identified which parameter regions work well for this stock.
- **Cycle 30** (day 15): 1,000+ trials. Model has converged — further trials only make marginal improvements. The bot notices this and reduces search to exploitation mode.

Each stock has its own isolated Optuna study. TCS's optimal parameters have nothing to do with PVRINOX's — they're learned independently.

---

## The Models

### Daily Ensemble

| Component | Role | Weight |
|-----------|------|--------|
| XGBoost Classifier | Predicts direction (up/down) | 55% |
| XGBoost Regressor | Predicts return magnitude (log-return) | — |
| Bidirectional LSTM | Captures multi-day sequential patterns | 45% |

The ensemble blends XGBoost and LSTM probabilities: `P(up) = 0.55 × xgb_prob + 0.45 × lstm_prob`. The regressor's output converts direction + magnitude into an actual predicted price (e.g. "up 0.8% → ₹3,847").

**Walk-forward cross-validation** prevents any lookahead bias. The model is never evaluated on data it trained on.

### Intraday Model

A lightweight XGBoost classifier + regressor trained specifically on 10-minute bars. Designed to retrain in under 500ms so it can update every single bar without slowing anything down. Uses 20+ intraday-specific features (bar shape, VWAP deviation, time-of-day, volume anomalies) that don't exist in daily data.

### Optuna Hyperparameter Search

**16 parameters searched per stock:**

| Category | Parameters |
|----------|-----------|
| Boosting schedule | `n_estimators` (80–800), `learning_rate` (0.005–0.4) |
| Tree structure | `max_depth` (2–10), `min_child_weight` (1–50), `gamma` (0–5), `max_delta_step` (0–10) |
| Regularisation | `reg_alpha`, `reg_lambda` (both log-scale) |
| Feature sampling | `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` (0.3–1.0) |
| Row sampling | `subsample` (0.4–1.0) |
| Tree policy | `grow_policy` (depthwise / lossguide), `max_leaves` |
| Class balance | `scale_pos_weight` (0.5–2.0) |

The `lossguide` grow policy is LightGBM-style leaf-wise splitting — for some stocks it outperforms the standard depth-wise approach by a significant margin. Optuna discovers which is better per ticker.

---

## Features Explained

### Daily model — ~60 features

**Technical indicators**
RSI(14), MACD signal/histogram, Bollinger Band position, ATR, EMA crossovers (8/21/50/200), OBV, Stochastic %K/%D, ADX, Williams %R, CCI.

**Price momentum**
Log returns at 1/5/10/20/60 day horizons. Rolling volatility. 52-week high/low distance.

**Institutional flow**
FII net buy/sell (today + 5-day rolling), DII net buy/sell (today + 5-day rolling), combined FII+DII net. Sourced from NSE's official API.

**Delivery %**
Daily equity delivery percentage from NSE bhav copy archives. High delivery = real money moving, not intraday speculation. Includes 5-day rolling average and a spike flag (>1.5× 20-day average).

**Sector rotation**
5-day and 20-day return of the stock's Nifty sector index. Rolling percentile rank (sector momentum). Whether the sector index is above its 50-day MA. Stocks in strong sectors tend to outperform regardless of company-specific news.

**Sentiment**
Daily average FinBERT score across all news for that stock. Number of bullish/bearish articles. Separate scores for company news vs macro news. NSE announcement flag (results, dividends, board meetings).

**Calendar**
F&O expiry week flag, last trading day of month, earnings season flag (Apr/Jul/Oct/Jan), budget week flag.

### Intraday model — ~25 features

Log returns over last 1/2/3/5/10 bars, bar body % (open-to-close / high-to-low), upper and lower wick %, bull bar flag, volume ratio vs 10-bar average, volume z-score, VWAP deviation, RSI(7), EMA-5 vs EMA-15 crossover and deviation, cumulative intraday return from open, ATR ratio (5-bar vs 20-bar), bars since open, hour of day, morning/afternoon/last-hour flags, 10-bar linear regression slope.

---

## Requirements & Setup

### Requirements

| Tool | Minimum |
|------|---------|
| Python | 3.11 |
| Node.js | 18 |

### First-time setup

**macOS / Linux:**
```bash
git clone <repo-url>
cd nse-simulator
chmod +x start.sh
./start.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Both launchers create the Python venv, install all dependencies, and start both servers. Open **http://localhost:3000**.

> **Note:** The first time you add a stock, FinBERT (~500 MB) downloads automatically. Give it 2–5 minutes.

### Environment variables

Copy `backend/.env.example` to `backend/.env` and fill in:

```bash
# Optional — for news fetching
NEWSAPI_KEY=your_key_here

# Optional — for Reddit sentiment
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# Optional — for cloud DB sync across machines (see below)
DATABASE_URL=postgresql://...
```

---

## Running Everything

You need **two terminals** running at the same time.

**Terminal 1 — Main server:**
```bash
# macOS / Linux
./start.sh

# Windows
powershell -ExecutionPolicy Bypass -File start.ps1
```

**Terminal 2 — Auto-bot:**
```bash
# macOS / Linux
cd backend
source .venv/bin/activate
python auto_trainer.py

# Windows
cd backend
.venv\Scripts\Activate.ps1
python auto_trainer.py
```

The main server handles the web UI and API. The auto-bot handles all training and prediction loops. They communicate over the local API — the bot calls the same endpoints the browser does.

You can run without the auto-bot (the web UI still works, you just won't get intraday predictions or automatic overnight retrains). The bot is what makes the system self-improving.

---

## The Dashboard

### Watchlist (home page)

Lists all tracked stocks with their current model accuracy, last refresh time, and a quick signal indicator. Add any NSE ticker by typing it in the search bar — the system validates it and starts the pipeline automatically.

### Stock detail page

**Live price panel** — shows real-time price, open/high/low, previous close, and change from open. Toggle **Live Mode** to poll every 5/10/30/60 seconds.

**Model vs Reality box** — appears in Live Mode. Shows what the model predicted for today (up/down, target price) vs what the stock is actually doing. Turns green when converging, orange when diverging.

**Stats row:**
- **Model Accuracy** — walk-forward CV accuracy (honest out-of-sample, not in-sample)
- **Sharpe Ratio** — risk-adjusted return of the model's trading strategy in backtest
- **Model Return** — cumulative return of following the model's signals in backtest
- **Buy & Hold** — what you'd have made just holding the stock
- **Alpha** — model return minus buy & hold (the model's actual edge)
- **Max Drawdown** — worst peak-to-trough loss in the backtest
- **Trading Days** — number of days in the backtest period

**Price chart** has six tabs:

| Tab | What it shows |
|-----|--------------|
| **Today** | 10-minute intraday bars with live predictions. Green dots = correct past predictions, red = wrong. Orange dotted line = model's next-bar prediction. Auto-refreshes every 30s. |
| **1M / 3M / 6M / 1Y / ALL** | Daily closing prices with model's historical predictions, initial vs updated prediction lines, confidence bands, buy/sell signal dots, sentiment overlay, and 30-day forward forecast. |

**Tomorrow's prediction card** — shown when Live Mode is off. Displays the ensemble's precise price estimate for tomorrow with confidence bands (XGBoost + LSTM contributions shown separately).

**Recent news** — all scraped headlines with FinBERT sentiment labels, source, and age.

---

## Cloud Database Sync

By default the system stores everything in a local SQLite file. To sync data between machines (e.g. your Mac and a Windows PC), point both to the same cloud PostgreSQL database.

**Recommended: Supabase (free, no credit card needed)**

1. Go to [supabase.com](https://supabase.com) → create a new project.
2. **Settings → Database → Connection string** → copy the **Transaction pooler** URL (not the direct connection URL — the direct one may not resolve from outside Supabase's network).
3. Paste it into `backend/.env`:

```bash
DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
```

4. Do the same on your other machine's `.env`.
5. Restart the backend — it automatically creates all tables on startup.

**What syncs:** stocks, predictions, news, portfolio holdings, sentiment scores.

**What stays local:** ML model weights (`.joblib`), LSTM weights (`.pt`), Optuna studies (`.pkl`), intraday session logs. These are machine-specific and too large for a database anyway.

---

## Architecture

```
nse-simulator/
├── backend/                        FastAPI (Python 3.11)
│   ├── main.py                     All 20+ API routes
│   ├── auto_trainer.py             Self-improving training bot
│   ├── config.py                   Env vars, paths, constants
│   │
│   ├── data/
│   │   ├── price_fetcher.py        5-year daily OHLCV (yfinance)
│   │   ├── intraday_fetcher.py     10-min bars (yfinance)
│   │   ├── live_price_fetcher.py   Real-time price (yf.fast_info)
│   │   ├── fii_dii_fetcher.py      Institutional flow (NSE API)
│   │   ├── delivery_fetcher.py     Delivery % (NSE bhav copy)
│   │   ├── news_scraper.py         Indian financial news sites
│   │   ├── macro_news.py           Macro headlines (NewsAPI)
│   │   ├── nse_announcements.py    Corporate filings (NSE)
│   │   └── reddit_fetcher.py       Reddit sentiment (PRAW)
│   │
│   ├── features/
│   │   ├── feature_builder.py      Master feature pipeline (~60 features)
│   │   ├── sector_rotation.py      Nifty sector index signals
│   │   ├── sentiment.py            FinBERT scoring + aggregation
│   │   └── calendar_flags.py       Expiry, earnings, budget flags
│   │
│   ├── model/
│   │   ├── train.py                Walk-forward CV + XGBoost + LSTM training
│   │   ├── intraday_trainer.py     10-min model (<500ms retrain)
│   │   ├── predict.py              Daily ensemble prediction + 30-day forecast
│   │   ├── tune.py                 Optuna 16-param TPE search
│   │   ├── lstm_model.py           Bidirectional 2-layer LSTM (PyTorch)
│   │   ├── backtest.py             OOS portfolio simulation
│   │   └── signals.py              Buy/sell signal generation
│   │
│   └── db/
│       ├── models.py               SQLAlchemy ORM (Stock, Prediction, News, Portfolio)
│       └── database.py             Engine + migrations (SQLite / PostgreSQL)
│
└── frontend/                       Next.js + TypeScript
    ├── pages/
    │   ├── index.tsx               Watchlist + portfolio overview
    │   └── stock/[ticker].tsx      Stock detail + Live Mode + Tomorrow card
    ├── components/
    │   └── StockChart.tsx          Today (intraday) + historical chart tabs
    └── lib/
        └── api.ts                  Typed API client
```

---

## API Reference

Full interactive docs at **http://localhost:8000/docs**.

### Stocks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List all tracked stocks with metadata |
| POST | `/api/stocks` | Add stock `{"ticker": "TCS"}` |
| DELETE | `/api/stocks/{ticker}` | Remove stock and all its data |
| POST | `/api/stocks/{ticker}/refresh` | Re-run full pipeline + Optuna |
| GET | `/api/stocks/{ticker}/chart` | Historical predictions + 30-day forecast |
| GET | `/api/stocks/{ticker}/stats` | Backtest stats (Sharpe, alpha, drawdown) |
| GET | `/api/stocks/{ticker}/news` | News with FinBERT sentiment |
| GET | `/api/stocks/{ticker}/live` | Live price + model vs reality comparison |

### Intraday

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/{ticker}/intraday` | Today's 10-min bars + prediction + session accuracy |
| POST | `/api/stocks/{ticker}/intraday/train` | Train intraday model on last 5 days |
| POST | `/api/stocks/{ticker}/intraday/predict` | Make next-bar prediction + record it |
| POST | `/api/stocks/{ticker}/intraday/record-actual` | Record actual price `{"actual_price": 986.50}` |
| POST | `/api/stocks/{ticker}/intraday/replay` | Offline sequential replay `?days_back=5` |

### Portfolio

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio` | All holdings with current P&L |
| POST | `/api/portfolio` | Add holding |
| PUT | `/api/portfolio/{ticker}` | Update buy price / quantity |
| DELETE | `/api/portfolio/{ticker}` | Remove holding |

---

## Configuration

**`backend/auto_trainer.py` — bot config:**
```python
BACKEND_URL            = "http://localhost:8000"
INTRADAY_BAR_SECONDS   = 600   # 10-minute bars
INTRADAY_BAR_BUFFER    = 60    # extra wait after bar close (seconds)
SLEEP_BETWEEN_STOCKS   = 30    # pause between overnight pipeline runs
ADD_NEW_EVERY_N_CYCLES = 2     # discover new stocks every N overnight cycles
MAX_STOCKS             = 20    # cap on tracked stocks (raise to explore more)
```

**`backend/config.py` — model config:**
```python
WALK_FORWARD_WINDOW        = 252   # ~1 year of training minimum
SIGNAL_CONFIDENCE_THRESHOLD = 0.60  # minimum confidence for buy/sell signal
HISTORY_YEARS              = 5     # years of daily data to fetch
```

To reset a stock's Optuna tuning history (start fresh):
```bash
rm backend/model/saved/TCS_optuna_study.pkl
```

---

## Data Sources

| Source | What's fetched |
|--------|---------------|
| Yahoo Finance | 5-year daily OHLCV, 10-min intraday bars, live prices, Nifty sector indices |
| NSE India API | FII/DII net institutional flow, corporate announcements |
| NSE Bhav Copy | Daily equity delivery % |
| NewsAPI | Macro and company news headlines |
| Web scrapers | Moneycontrol, Economic Times, LiveMint, and other Indian financial sites |
| Reddit | r/IndiaInvestments — retail investor sentiment |

---

## Troubleshooting

**Port 8000 already in use:**
```bash
# macOS / Linux
lsof -ti:8000 | xargs kill -9

# Windows PowerShell
netstat -ano | findstr :8000
Stop-Process -Id <PID> -Force
```

**FinBERT slow to download** — one-time ~500 MB download on first stock add, cached at `~/.cache/huggingface`. Don't interrupt it.

**"Today" tab shows no bars** — market is closed (or it's a weekend/holiday), OR `auto_trainer.py` is not running. The tab auto-refreshes every 30s and will populate as soon as bars exist.

**Supabase connection fails** — use the **Transaction Pooler** URL from Supabase (Settings → Database → Connection string), not the direct connection URL. The direct URL's hostname (`db.xxx.supabase.co`) may not resolve from outside Supabase's network.

**FII/DII returns zeros** — NSE's API requires browser-like session cookies that expire. The fetcher retries up to 10 times and falls back to zeros silently. The model still runs normally.

**Model accuracy stuck at ~50%** — this is normal for volatile stocks. The model is reporting honest out-of-sample accuracy. Run the auto-bot overnight to accumulate Optuna trials — accuracy typically improves 1–3% over the first week.

---

## License

MIT
