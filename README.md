# NSE Stock Simulator

A full-stack ML-powered trading simulator for NSE (National Stock Exchange of India) stocks. The system continuously trains and self-improves using Optuna hyperparameter search, provides intraday 10-minute bar predictions, and shows sentiment-aware buy/sell signals.

---

## Features

- **Daily model** — XGBoost classifier + regressor + LSTM ensemble predicting tomorrow's direction and price, trained on 5+ years of historical data.
- **Intraday model** — Lightweight XGBoost on 10-minute bars, retraining every bar during market hours.
- **Optuna tuning** — 16-parameter search space, per-ticker persistent studies that get smarter with every refresh.
- **Live mode** — Polls Yahoo Finance every 5–30 seconds for real-time prices; model vs actual comparison shown live.
- **Today chart** — New "Today" tab shows 10-minute bars with live predictions, accuracy markers, and 30-second auto-refresh.
- **Sentiment analysis** — FinBERT-scored news from multiple sources, overlaid on the chart.
- **Portfolio tracker** — Holdings with current P&L, model signals, and signals confidence.
- **Institutional flow** — FII/DII data and delivery % as model features.
- **Sector rotation** — Nifty sector indices as relative-strength features.
- **Cross-platform** — macOS, Linux, and Windows.

---

## Requirements

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| Node.js | 18 |

---

## Quick Start

### macOS / Linux

```bash
git clone <repo-url>
cd nse-simulator

chmod +x start.sh
./start.sh
```

### Windows (PowerShell)

```powershell
# From the project root:
powershell -ExecutionPolicy Bypass -File start.ps1
```

Both launchers:
1. Create a Python virtual environment and install all backend dependencies.
2. Install Node.js packages for the frontend.
3. Start the FastAPI backend at **http://localhost:8000**.
4. Start the Next.js frontend at **http://localhost:3000**.

Open **http://localhost:3000** in your browser and add your first NSE ticker.

> **First run:** Adding a stock triggers a one-time download of the FinBERT model (~500 MB). This takes 2–5 minutes on most connections.

---

## Running the Auto-Trainer

The auto-trainer runs indefinitely, handling both intraday live training and overnight Optuna retrains. Run it in a separate terminal alongside the main server.

### macOS / Linux

```bash
cd backend
source .venv/bin/activate
python auto_trainer.py
```

### Windows (PowerShell)

```powershell
cd backend
.venv\Scripts\Activate.ps1
python auto_trainer.py
```

### What it does

**Market hours (9:15 AM – 3:30 PM IST, Mon–Fri) — 10-minute intraday loop:**
1. Trains the intraday XGBoost model on the last 5 trading days of 10-minute bars.
2. Predicts the direction and price of the next 10-minute bar for every tracked stock.
3. Waits ~10 minutes for the bar to close.
4. Fetches actual prices, records whether the prediction was correct, and prints a running accuracy table.
5. Retrains on the updated bars.
6. Repeats until market closes.

**After market close — offline replay + overnight retrain:**
1. Replays the last 5 days of 10-minute bars sequentially (predict → check → retrain → advance) to measure how accurate the intraday model would have been.
2. Triggers a full Optuna hyperparameter search for every tracked stock (35–60 trials per stock).
3. Optionally adds a random stock from the NSE universe.
4. Sleeps until 9:10 AM IST on the next trading day.

Press **Ctrl+C** to stop cleanly. All models and Optuna studies are saved automatically.

---

## Architecture

```
nse-simulator/
├── backend/                       FastAPI (Python 3.11)
│   ├── main.py                    All API routes
│   ├── auto_trainer.py            Continuous training loop
│   ├── config.py                  Paths, thresholds, constants
│   ├── data/
│   │   ├── price_fetcher.py       Daily OHLCV from yfinance
│   │   ├── intraday_fetcher.py    10-min bars from yfinance
│   │   ├── live_price_fetcher.py  Live price via yf.fast_info
│   │   ├── fii_dii_fetcher.py     FII/DII flow from NSE API
│   │   ├── delivery_fetcher.py    Delivery % from NSE bhav copy
│   │   ├── news_scraper.py        Web news scraping
│   │   └── macro_news.py          NewsAPI macro headlines
│   ├── features/
│   │   ├── feature_builder.py     Full daily feature pipeline
│   │   ├── sector_rotation.py     Nifty sector index signals
│   │   ├── sentiment.py           FinBERT scoring
│   │   └── calendar_flags.py      Expiry / earnings calendar
│   ├── model/
│   │   ├── train.py               Walk-forward CV + XGBoost + LSTM
│   │   ├── intraday_trainer.py    10-min XGBoost (< 500 ms retrain)
│   │   ├── predict.py             Daily ensemble prediction
│   │   ├── tune.py                Optuna 16-param search
│   │   ├── lstm_model.py          Bidirectional LSTM (PyTorch)
│   │   ├── backtest.py            OOS portfolio simulation
│   │   └── signals.py             Buy/sell signal generation
│   └── db/
│       ├── models.py              SQLAlchemy ORM models
│       └── database.py            DB init + schema migrations
└── frontend/                      Next.js + TypeScript
    ├── pages/
    │   ├── index.tsx              Watchlist + portfolio dashboard
    │   └── stock/[ticker].tsx     Per-stock detail page + Live Mode
    ├── components/
    │   └── StockChart.tsx         Recharts (Today / 1M / 3M / 6M / 1Y / All)
    └── lib/
        └── api.ts                 API client + TypeScript interfaces
```

### Model stack

| Component | Purpose |
|-----------|---------|
| XGBoost Classifier | Next-day direction (primary signal) |
| XGBoost Regressor | Next-day price magnitude (log-return) |
| Bidirectional LSTM | Sequential pattern recognition |
| Ensemble | 0.55 × XGB + 0.45 × LSTM probability blend |
| Intraday XGBoost | 10-min direction + price (fast retrain) |
| Optuna TPE | Per-ticker hyperparameter optimisation (16 params) |

### Daily model features

Technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA crossovers, OBV, Stochastic, ADX), price momentum at 5/10/20/60-day horizons, FII/DII net flow (5-day rolling), equity delivery %, sector index relative strength (5d/20d), FinBERT sentiment scores, NSE corporate announcements, calendar flags (expiry week, earnings, IPO windows).

### Intraday model features

1/2/3/5/10-bar log returns, bar shape (body %, upper/lower wick %, bull/bear flag), volume ratio and z-score, VWAP deviation, RSI-7, EMA-5 vs EMA-15 crossover + deviation, cumulative intraday return from open, ATR ratio (5-bar / 20-bar), time-of-day (bars since open, hour, morning/afternoon/last-hour flags), 10-bar linear regression slope.

---

## API Reference

Full interactive docs at **http://localhost:8000/docs**.

### Stocks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List all tracked stocks |
| POST | `/api/stocks` | Add stock `{"ticker": "TCS"}` |
| DELETE | `/api/stocks/{ticker}` | Remove stock |
| POST | `/api/stocks/{ticker}/refresh` | Retrain + Optuna search |
| GET | `/api/stocks/{ticker}/chart` | Historical + 30-day forecast data |
| GET | `/api/stocks/{ticker}/stats` | Backtest stats (Sharpe, alpha, drawdown) |
| GET | `/api/stocks/{ticker}/news` | News with FinBERT sentiment scores |
| GET | `/api/stocks/{ticker}/live` | Live price + model comparison |

### Intraday (10-minute)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/{ticker}/intraday` | Today's bars + current prediction + session accuracy |
| POST | `/api/stocks/{ticker}/intraday/train` | Train intraday model on 5-day bars |
| POST | `/api/stocks/{ticker}/intraday/predict` | Predict next bar + log in session |
| POST | `/api/stocks/{ticker}/intraday/record-actual` | Record actual `{"actual_price": 3812.50}` |
| POST | `/api/stocks/{ticker}/intraday/replay` | Offline sequential replay (`?days_back=5`) |

### Portfolio

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio` | All holdings with P&L |
| POST | `/api/portfolio` | Add holding |
| PUT | `/api/portfolio/{ticker}` | Update buy price / quantity |
| DELETE | `/api/portfolio/{ticker}` | Remove holding |

---

## Configuration

Edit the `CONFIG` block at the top of `backend/auto_trainer.py`:

```python
BACKEND_URL           = "http://localhost:8000"
INTRADAY_BAR_SECONDS  = 600    # bar length in seconds (10 min)
INTRADAY_BAR_BUFFER   = 60     # extra wait after bar close
SLEEP_BETWEEN_STOCKS  = 30     # gap between overnight pipeline runs
ADD_NEW_EVERY_N_CYCLES = 2     # add random stock every N overnight cycles
MAX_STOCKS            = 20     # cap on total tracked stocks
```

Edit `backend/config.py` for model-level settings (walk-forward window, signal confidence threshold, database path, etc.).

---

## Optuna Tuning Details

The 16-parameter search space covers:

- **Boosting schedule** — `n_estimators` (80–800), `learning_rate` (0.005–0.4 log)
- **Tree structure** — `max_depth` (2–10), `min_child_weight` (1–50), `max_delta_step` (0–10), `gamma` (0–5)
- **Regularisation** — `reg_alpha` (log), `reg_lambda` (log)
- **Sampling** — `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` (all 0.3–1.0)
- **Grow policy** — depthwise vs lossguide, `max_leaves` (lossguide only)
- **Class weighting** — `scale_pos_weight` (0.5–2.0)

Studies are saved to `backend/model/saved/<TICKER>_optuna_study.pkl`. To reset a ticker's tuning history:

```bash
rm backend/model/saved/TCS_optuna_study.pkl
```

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

**FinBERT download stuck** — Be patient on first run. The ~500 MB model is a one-time download cached to `~/.cache/huggingface`.

**No data in "Today" tab** — The intraday model needs `auto_trainer.py` running, or at least one call to `POST /intraday/train`. The tab works best during NSE market hours (9:15 AM – 3:30 PM IST, Mon–Fri).

**Yahoo Finance rate limiting** — Heavy polling can trigger temporary 429 errors. The system retries with backoff and falls back gracefully.

**FII/DII data unavailable** — NSE's API requires session cookies that expire periodically. After 10 consecutive failures the fetcher returns zeros and the model runs normally.

---

## Data Sources

| Source | Data fetched |
|--------|-------------|
| Yahoo Finance (yfinance) | Daily + intraday OHLCV, live prices, sector indices |
| NSE India API | FII/DII institutional flow, corporate announcements |
| NSE Bhav Copy Archives | Daily equity delivery percentage |
| NewsAPI | Macro and company-specific news headlines |
| Web scrapers | Indian financial news sites |
| Reddit (r/IndiaInvestments) | Retail investor sentiment |

---

## License

MIT
