import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import Link from "next/link";
import { Plus, RefreshCw, TrendingUp, BarChart3, Briefcase } from "lucide-react";
import StockCard from "../components/StockCard";
import AddStockModal from "../components/AddStockModal";
import { fetchStocks, removeStock, StockSummary } from "../lib/api";

export default function Home() {
  const [stocks, setStocks] = useState<StockSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const loadStocks = useCallback(async () => {
    try {
      const data = await fetchStocks();
      setStocks(data);
      setError(null);
    } catch (err: any) {
      setError(err?.message || "Failed to connect to backend. Is it running on port 8000?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStocks();
    // Poll every 15s to pick up background pipeline completions
    const interval = setInterval(() => loadStocks(), 15000);
    return () => clearInterval(interval);
  }, [loadStocks]);

  const handleRemove = async (ticker: string) => {
    if (!confirm(`Remove ${ticker} from tracking?`)) return;
    try {
      await removeStock(ticker);
      setStocks((prev) => prev.filter((s) => s.ticker !== ticker));
    } catch (err: any) {
      alert(`Failed to remove ${ticker}: ${err?.message}`);
    }
  };

  const handleRefreshAll = async () => {
    setRefreshing(true);
    await loadStocks();
    setRefreshing(false);
  };

  const processingCount = stocks.filter((s) => !s.last_updated).length;

  return (
    <>
      <Head>
        <title>NSE Simulator</title>
        <meta name="description" content="NSE stock prediction simulator with ML and sentiment analysis" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <TrendingUp size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">NSE Simulator</h1>
                <p className="text-xs text-slate-500">ML-powered NSE stock predictor</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Link
                href="/portfolio"
                className="btn-secondary flex items-center gap-2 text-sm"
              >
                <Briefcase size={14} />
                My Portfolio
              </Link>
              {processingCount > 0 && (
                <span className="text-xs text-yellow-400 flex items-center gap-1.5 mr-2">
                  <RefreshCw size={12} className="animate-spin" />
                  {processingCount} stock{processingCount !== 1 ? "s" : ""} processing…
                </span>
              )}
              <button
                onClick={handleRefreshAll}
                disabled={refreshing}
                className="btn-secondary flex items-center gap-2 text-sm"
                title="Refresh stock list"
              >
                <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
                Refresh
              </button>
              <button
                onClick={() => setIsModalOpen(true)}
                className="btn-primary flex items-center gap-2 text-sm"
              >
                <Plus size={14} />
                Add Stock
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Error state */}
          {error && (
            <div className="mb-6 bg-red-900/20 border border-red-700/40 rounded-xl p-4 text-red-400 text-sm">
              <strong>Connection Error:</strong> {error}
            </div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="card animate-pulse">
                  <div className="h-4 bg-slate-700 rounded w-20 mb-3" />
                  <div className="h-3 bg-slate-700 rounded w-32 mb-4" />
                  <div className="h-8 bg-slate-700 rounded w-24 mb-4" />
                  <div className="h-3 bg-slate-700 rounded w-full" />
                </div>
              ))}
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && stocks.length === 0 && (
            <div className="text-center py-24">
              <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <BarChart3 size={32} className="text-slate-600" />
              </div>
              <h2 className="text-lg font-semibold text-white mb-2">No stocks yet</h2>
              <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">
                Add your first NSE stock to get started. The simulator will fetch historical data,
                score news sentiment with FinBERT, and train an XGBoost prediction model.
              </p>
              <button
                onClick={() => setIsModalOpen(true)}
                className="btn-primary inline-flex items-center gap-2"
              >
                <Plus size={16} />
                Add your first stock
              </button>

              <div className="mt-12 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto text-left">
                {[
                  {
                    icon: "📈",
                    title: "Price Prediction",
                    desc: "XGBoost model trained on 5 years of OHLCV data with walk-forward validation",
                  },
                  {
                    icon: "📰",
                    title: "Sentiment Analysis",
                    desc: "FinBERT scores news from Economic Times, Moneycontrol, Reddit, and NewsAPI",
                  },
                  {
                    icon: "🧪",
                    title: "Backtesting",
                    desc: "Full portfolio simulation with Sharpe ratio, max drawdown, and buy/sell signals",
                  },
                ].map((feature) => (
                  <div key={feature.title} className="card text-center">
                    <div className="text-2xl mb-2">{feature.icon}</div>
                    <h3 className="text-sm font-semibold text-white mb-1">{feature.title}</h3>
                    <p className="text-xs text-slate-400">{feature.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stock grid */}
          {!loading && stocks.length > 0 && (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-medium text-slate-400">
                  Tracking {stocks.length} stock{stocks.length !== 1 ? "s" : ""}
                </h2>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {stocks.map((stock) => (
                  <StockCard key={stock.ticker} stock={stock} onRemove={handleRemove} />
                ))}
              </div>
            </>
          )}
        </main>
      </div>

      <AddStockModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAdded={loadStocks}
      />
    </>
  );
}
