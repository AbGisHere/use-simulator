import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import Link from "next/link";
import {
  ArrowLeft,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  BarChart2,
  ExternalLink,
  AlertCircle,
  Loader2,
  Briefcase,
} from "lucide-react";
import {
  ChartResponse,
  StockStats,
  NewsResponse,
  NewsItem,
  TomorrowPrediction,
  LiveData,
  fetchChartData,
  fetchStats,
  fetchNews,
  refreshStock,
  fetchLivePrice,
} from "../../lib/api";
import StockChart from "../../components/StockChart";
import SentimentOverlay from "../../components/SentimentOverlay";
import { formatDistanceToNow, parseISO } from "date-fns";

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="card">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className={`text-xl font-bold ${color || "text-white"}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

function TomorrowCard({ pred, currentPrice }: { pred: TomorrowPrediction; currentPrice: number | null }) {
  const isUp = pred.direction === 1;
  const ret  = pred.predicted_return_pct;
  const conf = Math.round(pred.confidence * 100);
  const fmt  = (n: number) => n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const hasLstm = pred.lstm_up_prob !== null;

  return (
    <div className={`rounded-xl border p-4 ${
      isUp
        ? "bg-green-950/40 border-green-700/40"
        : "bg-red-950/40 border-red-700/40"
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Tomorrow's Prediction
          </span>
          {hasLstm && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-300 border border-blue-700/40 font-mono">
              XGBoost + LSTM
            </span>
          )}
        </div>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
          isUp ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"
        }`}>
          {conf}% confident
        </span>
      </div>

      {/* Main price */}
      <div className="flex items-end gap-3 mb-3">
        <div>
          <span className={`text-3xl font-bold tracking-tight ${isUp ? "text-green-300" : "text-red-300"}`}>
            {isUp ? "↑" : "↓"} ₹{fmt(pred.predicted_price)}
          </span>
          <span className={`ml-2 text-sm font-medium ${isUp ? "text-green-400" : "text-red-400"}`}>
            {ret >= 0 ? "+" : ""}{ret.toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Price range band */}
      <div className="flex items-center gap-2 text-sm mb-3">
        <span className="text-slate-500 text-xs">Expected range:</span>
        <span className="font-mono text-slate-300 text-xs">
          ₹{fmt(pred.predicted_price_low)}
          <span className="text-slate-600 mx-1">–</span>
          ₹{fmt(pred.predicted_price_high)}
        </span>
      </div>

      {/* Visual confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>Bearish</span>
          <span>Bullish</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${isUp ? "bg-green-500" : "bg-red-500"}`}
            style={{ width: `${Math.round(pred.confidence * 100)}%`, marginLeft: isUp ? "auto" : undefined }}
          />
        </div>
        {currentPrice && (
          <p className="text-[10px] text-slate-600 text-right">
            From ₹{fmt(currentPrice)} → ₹{fmt(pred.predicted_price)}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Live Mode panel ───────────────────────────────────────────────────────────

const INTERVALS = [
  { label: "5s",  value: 5000 },
  { label: "10s", value: 10000 },
  { label: "30s", value: 30000 },
  { label: "60s", value: 60000 },
];

const STATUS_LABELS: Record<string, string> = {
  open:           "Market Open",
  pre_open:       "Pre-Open",
  pre_pre_open:   "Pre-Market",
  post_close:     "Post-Close",
  closed:         "Market Closed",
  closed_weekend: "Weekend",
};

function LiveModePanel({ ticker, onPriceUpdate }: {
  ticker: string;
  onPriceUpdate?: (price: number) => void;
}) {
  const [live, setLive] = useState<LiveData | null>(null);
  const [interval, setIntervalMs] = useState(10000);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [pulseKey, setPulseKey] = useState(0);
  const [loading, setLoading] = useState(true);

  const poll = useCallback(async () => {
    try {
      const data = await fetchLivePrice(ticker);
      setLive(data);
      setLastUpdated(new Date());
      setPulseKey(k => k + 1);
      if (data.price && onPriceUpdate) onPriceUpdate(data.price);
    } catch {
      /* silent — keep showing last data */
    } finally {
      setLoading(false);
    }
  }, [ticker, onPriceUpdate]);

  useEffect(() => {
    poll();
    const id = setInterval(poll, interval);
    return () => clearInterval(id);
  }, [poll, interval]);

  const fmt = (n: number | null | undefined, dec = 2) =>
    n != null ? n.toLocaleString("en-IN", { minimumFractionDigits: dec, maximumFractionDigits: dec }) : "—";

  const isUp   = (live?.change ?? 0) >= 0;
  const mc     = live?.model_comparison;
  const status = live?.market_status ?? "closed";

  return (
    <div className="rounded-xl border border-blue-700/40 bg-blue-950/30 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          {/* Pulsing live dot */}
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500" />
          </span>
          <span className="text-xs font-bold text-blue-300 uppercase tracking-widest">Live</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full border font-medium ${
            status === "open"
              ? "bg-green-900/40 text-green-300 border-green-700/40"
              : "bg-slate-800 text-slate-400 border-slate-700/40"
          }`}>
            {STATUS_LABELS[status] ?? status}
          </span>
        </div>

        {/* Interval picker */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-500 mr-1">Refresh:</span>
          {INTERVALS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setIntervalMs(opt.value)}
              className={`text-[10px] px-1.5 py-0.5 rounded font-mono transition-colors ${
                interval === opt.value
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {loading && !live ? (
        <div className="text-xs text-slate-500 animate-pulse">Fetching live price…</div>
      ) : live?.error ? (
        <div className="text-xs text-red-400">Error: {live.error}</div>
      ) : live ? (
        <>
          {/* Live price row */}
          <div className="flex items-end gap-4 flex-wrap">
            <div key={pulseKey} className="animate-[fadeIn_0.3s_ease]">
              <p className={`text-4xl font-bold tracking-tight ${isUp ? "text-green-300" : "text-red-300"}`}>
                ₹{fmt(live.price)}
              </p>
              <p className={`text-sm font-medium mt-0.5 ${isUp ? "text-green-400" : "text-red-400"}`}>
                {isUp ? "+" : ""}{fmt(live.change)} ({isUp ? "+" : ""}{fmt(live.change_pct)}%)
              </p>
            </div>

            {/* Intraday stats */}
            <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs pb-0.5">
              <div>
                <p className="text-slate-500">Open</p>
                <p className="text-slate-300 font-mono">₹{fmt(live.open)}</p>
              </div>
              <div>
                <p className="text-slate-500">High</p>
                <p className="text-green-400 font-mono">₹{fmt(live.day_high)}</p>
              </div>
              <div>
                <p className="text-slate-500">Low</p>
                <p className="text-red-400 font-mono">₹{fmt(live.day_low)}</p>
              </div>
              <div>
                <p className="text-slate-500">Prev Close</p>
                <p className="text-slate-300 font-mono">₹{fmt(live.prev_close)}</p>
              </div>
              <div>
                <p className="text-slate-500">From Open</p>
                <p className={`font-mono ${(live.change_from_open ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {(live.change_from_open ?? 0) >= 0 ? "+" : ""}{fmt(live.change_from_open)}%
                </p>
              </div>
              <div>
                <p className="text-slate-500">Updated</p>
                <p className="text-slate-400 font-mono text-[10px]">
                  {lastUpdated ? lastUpdated.toLocaleTimeString("en-IN", { hour12: false }) : "—"}
                </p>
              </div>
            </div>
          </div>

          {/* Model prediction vs reality tracker */}
          {mc ? (
            <div className={`rounded-lg p-3 border text-xs space-y-2 ${
              mc.prediction_correct
                ? "bg-green-950/40 border-green-700/30"
                : "bg-orange-950/30 border-orange-700/30"
            }`}>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-300">Model vs Reality</span>
                <span className={`font-bold px-2 py-0.5 rounded-full text-[10px] ${
                  mc.prediction_correct
                    ? "bg-green-900/60 text-green-300"
                    : "bg-orange-900/60 text-orange-300"
                }`}>
                  {mc.prediction_correct ? "✓ Tracking correctly" : "✗ Diverging"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <p className="text-slate-500">Model called</p>
                  <p className={`font-semibold ${mc.predicted_direction === 1 ? "text-green-400" : "text-red-400"}`}>
                    {mc.predicted_direction === 1 ? "↑ UP" : "↓ DOWN"}
                    <span className="text-slate-500 font-normal ml-1">({Math.round(mc.confidence * 100)}% conf)</span>
                  </p>
                </div>
                <div>
                  <p className="text-slate-500">Actual so far</p>
                  <p className={`font-semibold ${mc.actual_direction_now === 1 ? "text-green-400" : "text-red-400"}`}>
                    {mc.actual_direction_now === 1 ? "↑ UP" : "↓ DOWN"}
                  </p>
                </div>
                <div>
                  <p className="text-slate-500">Target price</p>
                  <p className="text-slate-300 font-mono">₹{fmt(mc.predicted_price)}</p>
                  <p className="text-slate-500 text-[10px]">
                    Gap: {mc.gap_to_target >= 0 ? "+" : ""}₹{fmt(mc.gap_to_target)}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1 text-[10px] text-slate-500 pt-1">
                <span>Target range:</span>
                <span className="font-mono text-slate-400">₹{fmt(mc.predicted_price_low)} – ₹{fmt(mc.predicted_price_high)}</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">
              Run a Refresh Data first to generate model predictions for comparison.
            </p>
          )}
        </>
      ) : null}
    </div>
  );
}

function SentimentBadge({ label }: { label: string | null }) {
  if (!label) return <span className="badge-gray">neutral</span>;
  if (label === "positive") return <span className="badge-green">bullish</span>;
  if (label === "negative") return <span className="badge-red">bearish</span>;
  return <span className="badge-gray">neutral</span>;
}

function NewsCard({ item }: { item: NewsItem }) {
  const timeAgo = item.published_at
    ? formatDistanceToNow(parseISO(item.published_at), { addSuffix: true })
    : "";

  return (
    <div className="border-b border-slate-700 pb-3 mb-3 last:border-0 last:mb-0 last:pb-0">
      <div className="flex items-start justify-between gap-2 mb-1">
        <p className="text-sm text-slate-200 leading-snug line-clamp-2">{item.headline}</p>
        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 text-slate-600 hover:text-blue-400 transition-colors"
          >
            <ExternalLink size={13} />
          </a>
        )}
      </div>
      <div className="flex items-center gap-2">
        <SentimentBadge label={item.sentiment_label} />
        <span className="text-xs text-slate-500">{item.source}</span>
        {timeAgo && <span className="text-xs text-slate-600">{timeAgo}</span>}
      </div>
    </div>
  );
}

export default function StockPage() {
  const router = useRouter();
  const { ticker, buyPrice: buyPriceParam, buyQuantity: buyQuantityParam } = router.query as {
    ticker?: string;
    buyPrice?: string;
    buyQuantity?: string;
  };
  const buyPrice = buyPriceParam ? parseFloat(buyPriceParam) : undefined;
  const buyQuantity = buyQuantityParam ? parseFloat(buyQuantityParam) : undefined;

  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [stats, setStats] = useState<StockStats | null>(null);
  const [news, setNews] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [livePriceOverride, setLivePriceOverride] = useState<number | null>(null);

  const loadData = useCallback(async (t: string) => {
    try {
      const [chartRes, newsRes] = await Promise.all([
        fetchChartData(t),
        fetchNews(t),
      ]);
      setChartData(chartRes);
      setNews(newsRes);

      // Stats can fail if backtest hasn't run yet
      try {
        const statsRes = await fetchStats(t);
        setStats(statsRes);
      } catch {
        // Stats not ready yet
      }

      setError(null);
    } catch (err: any) {
      const detail = err?.response?.data?.detail || err?.message;
      setError(detail || "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    loadData(ticker);
  }, [ticker, loadData]);

  const handleRefresh = async () => {
    if (!ticker) return;
    setRefreshing(true);
    try {
      await refreshStock(ticker);
      alert("Refresh started! Check back in 2-3 minutes.");
    } catch (err: any) {
      alert(`Refresh failed: ${err?.message}`);
    } finally {
      setRefreshing(false);
    }
  };

  const latestData = chartData?.chart_data?.[chartData.chart_data.length - 1];
  const latestPrice = latestData?.actual_price;
  const prevData = chartData?.chart_data?.[chartData.chart_data.length - 2];
  const priceChange = latestPrice && prevData?.actual_price
    ? latestPrice - prevData.actual_price
    : null;
  const priceChangePct = priceChange && prevData?.actual_price
    ? (priceChange / prevData.actual_price) * 100
    : null;

  if (!ticker) return null;

  return (
    <>
      <Head>
        <title>{ticker} — NSE Simulator</title>
      </Head>

      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link href="/" className="flex items-center gap-1.5 text-slate-400 hover:text-white transition-colors text-sm">
                <ArrowLeft size={16} />
                Back
              </Link>
              <span className="text-slate-700">|</span>
              <span className="font-mono font-bold text-blue-400">{ticker}</span>
              {chartData?.display_name && (
                <span className="text-slate-400 text-sm hidden sm:block">{chartData.display_name}</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Link href="/portfolio" className="btn-secondary flex items-center gap-2 text-sm">
                <Briefcase size={14} />
                Portfolio
              </Link>
              <button
                onClick={() => setLiveMode(m => !m)}
                className={`flex items-center gap-2 text-sm px-3 py-1.5 rounded-lg border font-medium transition-all ${
                  liveMode
                    ? "bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/30"
                    : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700"
                }`}
              >
                {liveMode && (
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-200 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                  </span>
                )}
                {liveMode ? "Live ON" : "Live Mode"}
              </button>
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="btn-secondary flex items-center gap-2 text-sm"
              >
                <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
                Refresh Data
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Loading */}
          {loading && (
            <div className="flex items-center justify-center py-24 text-slate-400">
              <Loader2 size={24} className="animate-spin mr-3" />
              Loading {ticker} data…
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="bg-red-900/20 border border-red-700/40 rounded-xl p-6 text-center">
              <AlertCircle size={24} className="text-red-400 mx-auto mb-3" />
              <p className="text-red-400 font-medium mb-1">Failed to load data</p>
              <p className="text-red-300/70 text-sm">{error}</p>
              {error.includes("Pipeline may still be running") && (
                <p className="text-slate-400 text-xs mt-3">
                  The pipeline is running in the background. This page will auto-refresh.
                </p>
              )}
            </div>
          )}

          {/* Content */}
          {!loading && !error && chartData && (
            <div className="space-y-6">
              {/* Top section: price + stats */}
              <div>
                <div className="flex items-end gap-4 mb-4">
                  <div>
                    <p className="text-4xl font-bold text-white">
                      {latestPrice
                        ? `₹${latestPrice.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                        : "—"}
                    </p>
                    {priceChange != null && priceChangePct != null && (
                      <div className={`flex items-center gap-1.5 mt-1 ${priceChange >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {priceChange >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                        <span className="text-sm font-medium">
                          {priceChange >= 0 ? "+" : ""}
                          {priceChange.toFixed(2)} ({priceChangePct >= 0 ? "+" : ""}{priceChangePct.toFixed(2)}%)
                        </span>
                        <span className="text-slate-500 text-xs">vs prev day</span>
                      </div>
                    )}
                  </div>
                  {latestData?.signal && latestData.signal !== "hold" && (
                    <div className={`px-3 py-1.5 rounded-lg text-sm font-bold flex items-center gap-1.5 ${
                      latestData.signal === "buy"
                        ? "bg-green-900/40 text-green-400 border border-green-700/50"
                        : "bg-red-900/40 text-red-400 border border-red-700/50"
                    }`}>
                      {latestData.signal === "buy" ? "↑ BUY SIGNAL" : "↓ SELL SIGNAL"}
                      {latestData.confidence && (
                        <span className="font-normal text-xs opacity-70">
                          ({(latestData.confidence * 100).toFixed(0)}% conf)
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Live Mode panel */}
                {liveMode && (
                  <div className="mt-4">
                    <LiveModePanel
                      ticker={ticker}
                      onPriceUpdate={setLivePriceOverride}
                    />
                  </div>
                )}

                {/* Tomorrow's price prediction (hidden during live mode to avoid clutter) */}
                {!liveMode && chartData.tomorrow_prediction && (
                  <div className="mt-4 mb-2">
                    <TomorrowCard
                      pred={chartData.tomorrow_prediction}
                      currentPrice={livePriceOverride ?? latestPrice ?? null}
                    />
                  </div>
                )}

                {/* Stats row */}
                {stats && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
                    <StatCard
                      label="Model Accuracy"
                      value={`${stats.model_accuracy.toFixed(1)}%`}
                      color={stats.model_accuracy >= 55 ? "text-green-400" : stats.model_accuracy >= 50 ? "text-yellow-400" : "text-red-400"}
                    />
                    <StatCard
                      label="Sharpe Ratio"
                      value={stats.sharpe_ratio.toFixed(2)}
                      color={stats.sharpe_ratio > 1 ? "text-green-400" : stats.sharpe_ratio > 0 ? "text-yellow-400" : "text-red-400"}
                    />
                    <StatCard
                      label="Model Return"
                      value={`${stats.cumulative_return >= 0 ? "+" : ""}${stats.cumulative_return.toFixed(1)}%`}
                      color={stats.cumulative_return >= 0 ? "text-green-400" : "text-red-400"}
                    />
                    <StatCard
                      label="Buy & Hold"
                      value={`${stats.benchmark_return >= 0 ? "+" : ""}${stats.benchmark_return.toFixed(1)}%`}
                      color={stats.benchmark_return >= 0 ? "text-green-400" : "text-red-400"}
                    />
                    <StatCard
                      label="Alpha"
                      value={`${stats.alpha >= 0 ? "+" : ""}${stats.alpha.toFixed(1)}%`}
                      color={stats.alpha >= 0 ? "text-green-400" : "text-red-400"}
                    />
                    <StatCard
                      label="Max Drawdown"
                      value={`-${stats.max_drawdown.toFixed(1)}%`}
                      color="text-red-400"
                    />
                    <StatCard
                      label="Trading Days"
                      value={stats.total_trading_days.toString()}
                      sub="in backtest"
                    />
                  </div>
                )}
              </div>

              {/* Main chart + news side panel */}
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Chart (3/4 width on large screens) */}
                <div className="lg:col-span-3 space-y-4">
                  <div className="card">
                    <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                      <BarChart2 size={16} className="text-blue-400" />
                      Price Chart & Predictions
                    </h2>
                    <StockChart
                      data={chartData.chart_data}
                      futureData={chartData.future_data ?? []}
                      ticker={ticker}
                      buyPrice={buyPrice}
                      buyQuantity={buyQuantity}
                    />
                  </div>

                  {/* Sentiment bar chart */}
                  <div className="card">
                    <h2 className="text-sm font-semibold text-slate-300 mb-3">
                      Daily Sentiment Score
                    </h2>
                    <p className="text-xs text-slate-500 mb-3">
                      Aggregated FinBERT sentiment from news and Reddit. Click a bar to see contributing headlines.
                    </p>
                    <SentimentOverlay
                      data={chartData.chart_data}
                      news={news?.news || []}
                    />
                  </div>
                </div>

                {/* News side panel (1/4 width) */}
                <div className="lg:col-span-1">
                  <div className="card h-full max-h-[700px] flex flex-col">
                    <h2 className="text-sm font-semibold text-slate-300 mb-4">
                      Recent News
                      {news?.news?.length ? (
                        <span className="ml-2 text-xs text-slate-500">({news.news.length})</span>
                      ) : null}
                    </h2>
                    <div className="overflow-y-auto flex-1 pr-1">
                      {!news?.news?.length ? (
                        <p className="text-sm text-slate-500 text-center py-8">No news cached yet</p>
                      ) : (
                        news.news.map((item, i) => <NewsCard key={i} item={item} />)
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Portfolio simulation chart */}
              {chartData.portfolio_data?.length > 0 && (
                <PortfolioChart data={chartData.portfolio_data} />
              )}
            </div>
          )}
        </main>
      </div>
    </>
  );
}

// ── Portfolio chart ────────────────────────────────────────────────────────────

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { PortfolioDataPoint } from "../../lib/api";

function PortfolioChart({ data }: { data: PortfolioDataPoint[] }) {
  const formatValue = (v: number) =>
    `₹${(v / 1000).toFixed(1)}k`;

  return (
    <div className="card">
      <h2 className="text-sm font-semibold text-slate-300 mb-4">
        Portfolio Simulation vs Buy &amp; Hold
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Starting capital: ₹1,00,000. Model trades on signals with 0.1% transaction cost.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#64748b" }}
            tickLine={false}
            axisLine={{ stroke: "#334155" }}
            interval="preserveStartEnd"
            minTickGap={80}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#64748b" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatValue}
            width={55}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`,
              name === "portfolio_value" ? "Model Portfolio" : "Buy & Hold",
            ]}
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <Legend
            formatter={(value) => (
              <span style={{ color: "#94a3b8", fontSize: "12px" }}>
                {value === "portfolio_value" ? "Model Portfolio" : "Buy & Hold"}
              </span>
            )}
          />
          <Line
            type="monotone"
            dataKey="portfolio_value"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="benchmark_value"
            stroke="#64748b"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
