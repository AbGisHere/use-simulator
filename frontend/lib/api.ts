import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// ── Types ─────────────────────────────────────────────────────────────────────

export interface StockSummary {
  ticker: string;
  display_name: string;
  added_at: string | null;
  last_updated: string | null;
  last_price: number | null;
  model_accuracy: number;
}

export interface ChartDataPoint {
  date: string;
  actual_price: number;
  predicted_direction: number | null;          // latest model's direction
  initial_predicted_direction: number | null;  // first model run's direction
  confidence: number | null;
  initial_confidence: number | null;
  upper_band: number | null;
  lower_band: number | null;
  signal: "buy" | "sell" | "hold";
  sentiment_score: number;
}

export interface FutureDataPoint {
  date: string;
  projected_price: number;
  predicted_direction: number;
  confidence: number;
  upper_band: number;
  lower_band: number;
}

export interface TomorrowPrediction {
  date: string;
  predicted_price: number;
  predicted_price_low: number;
  predicted_price_high: number;
  predicted_return_pct: number;
  direction: number;          // 1 = up, 0 = down
  confidence: number;
  lstm_up_prob: number | null;
}

export interface PortfolioDataPoint {
  date: string;
  portfolio_value: number;
  benchmark_value: number;
}

export interface ChartResponse {
  ticker: string;
  display_name: string;
  chart_data: ChartDataPoint[];
  future_data: FutureDataPoint[];
  portfolio_data: PortfolioDataPoint[];
  tomorrow_prediction: TomorrowPrediction | null;
}

export interface StockStats {
  ticker: string;
  cumulative_return: number;
  benchmark_return: number;
  alpha: number;
  sharpe_ratio: number;
  max_drawdown: number;
  model_accuracy: number;
  total_trading_days: number;
}

export interface NewsItem {
  headline: string;
  source: string;
  published_at: string | null;
  sentiment_score: number | null;
  sentiment_label: string | null;
  url: string;
}

export interface NewsResponse {
  ticker: string;
  news: NewsItem[];
}

export interface PortfolioHolding {
  ticker: string;
  buy_price: number;
  quantity: number;
  notes: string | null;
  added_at: string | null;
  current_price: number | null;
  current_value: number | null;
  cost_basis: number;
  pnl_abs: number | null;
  pnl_pct: number | null;
  current_signal: "buy" | "sell" | "hold";
  signal_confidence: number | null;
}

export interface LiveModelComparison {
  predicted_direction: number;
  predicted_price: number;
  predicted_price_low: number;
  predicted_price_high: number;
  predicted_return_pct: number;
  confidence: number;
  actual_direction_now: number;
  prediction_correct: boolean;
  gap_to_target: number;
}

export interface LiveData {
  ticker: string;
  price: number | null;
  prev_close: number | null;
  open: number | null;
  day_high: number | null;
  day_low: number | null;
  change: number | null;
  change_pct: number | null;
  change_from_open: number | null;
  volume: number | null;
  market_status: "open" | "pre_open" | "pre_pre_open" | "post_close" | "closed" | "closed_weekend";
  is_trading: boolean;
  timestamp: string;
  error: string | null;
  model_comparison: LiveModelComparison | null;
}

// ── API Functions ─────────────────────────────────────────────────────────────

export async function fetchStocks(): Promise<StockSummary[]> {
  const res = await api.get<StockSummary[]>("/api/stocks");
  return res.data;
}

export async function addStock(ticker: string): Promise<{ ticker: string; display_name: string; status: string; message: string }> {
  const res = await api.post("/api/stocks", { ticker });
  return res.data;
}

export async function removeStock(ticker: string): Promise<void> {
  await api.delete(`/api/stocks/${ticker}`);
}

export async function fetchChartData(ticker: string): Promise<ChartResponse> {
  const res = await api.get<ChartResponse>(`/api/stocks/${ticker}/chart`);
  return res.data;
}

export async function fetchStats(ticker: string): Promise<StockStats> {
  const res = await api.get<StockStats>(`/api/stocks/${ticker}/stats`);
  return res.data;
}

export async function fetchNews(ticker: string, limit = 50): Promise<NewsResponse> {
  const res = await api.get<NewsResponse>(`/api/stocks/${ticker}/news`, {
    params: { limit },
  });
  return res.data;
}

export async function refreshStock(ticker: string): Promise<{ message: string }> {
  const res = await api.post(`/api/stocks/${ticker}/refresh`);
  return res.data;
}

export async function fetchLivePrice(ticker: string): Promise<LiveData> {
  const res = await api.get<LiveData>(`/api/stocks/${ticker}/live`);
  return res.data;
}

// ── Portfolio ─────────────────────────────────────────────────────────────────

export async function fetchPortfolio(): Promise<PortfolioHolding[]> {
  const res = await api.get<PortfolioHolding[]>("/api/portfolio");
  return res.data;
}

export async function addHolding(
  ticker: string,
  buy_price: number,
  quantity: number,
  notes?: string,
): Promise<{ ticker: string; buy_price: number; quantity: number; message: string }> {
  const res = await api.post("/api/portfolio", { ticker, buy_price, quantity, notes });
  return res.data;
}

export async function updateHolding(
  ticker: string,
  buy_price: number,
  quantity: number,
  notes?: string,
): Promise<{ ticker: string; buy_price: number; quantity: number }> {
  const res = await api.put(`/api/portfolio/${ticker}`, { ticker, buy_price, quantity, notes });
  return res.data;
}

export async function removeHolding(ticker: string): Promise<void> {
  await api.delete(`/api/portfolio/${ticker}`);
}

// ── Intraday (10-min) ─────────────────────────────────────────────────────────

export interface IntradayBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IntradayPrediction {
  direction: number;           // 1 = up, 0 = down
  confidence: number;
  up_probability: number;
  predicted_price: number;
  predicted_return: number;
  upper_band: number;
  lower_band: number;
  last_price: number;
  next_time: string;           // "HH:MM"
}

export interface SessionPrediction {
  time: string;
  predicted_dir: number | null;
  predicted_price: number | null;
  actual_price: number | null;
  correct: boolean | null;
}

export interface SessionAccuracy {
  total: number;
  correct: number;
  accuracy: number | null;
  predictions: SessionPrediction[];
}

export interface IntradayData {
  ticker: string;
  bars: IntradayBar[];
  prediction: IntradayPrediction | null;
  session_accuracy: SessionAccuracy;
}

export async function fetchIntradayData(ticker: string): Promise<IntradayData> {
  const res = await api.get<IntradayData>(`/api/stocks/${ticker}/intraday`);
  return res.data;
}
