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
  fetchChartData,
  fetchStats,
  fetchNews,
  refreshStock,
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
