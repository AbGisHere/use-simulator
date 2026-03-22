import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import Link from "next/link";
import { Plus, TrendingUp, TrendingDown, Briefcase, ArrowLeft, RefreshCw } from "lucide-react";
import PortfolioHoldingCard from "../components/PortfolioHoldingCard";
import AddHoldingModal from "../components/AddHoldingModal";
import { fetchPortfolio, removeHolding, PortfolioHolding } from "../lib/api";

function SummaryCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: "green" | "red" | "neutral";
}) {
  const colorClass =
    color === "green"
      ? "text-green-400"
      : color === "red"
      ? "text-red-400"
      : "text-white";

  return (
    <div className="card">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-xl font-bold ${colorClass}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function Portfolio() {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editHolding, setEditHolding] = useState<PortfolioHolding | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadHoldings = useCallback(async () => {
    try {
      const data = await fetchPortfolio();
      setHoldings(data);
      setError(null);
    } catch (err: any) {
      setError(err?.message || "Failed to load portfolio. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHoldings();
    const interval = setInterval(loadHoldings, 15000);
    return () => clearInterval(interval);
  }, [loadHoldings]);

  const handleRemove = async (ticker: string) => {
    if (!confirm(`Remove ${ticker} from your portfolio?`)) return;
    try {
      await removeHolding(ticker);
      setHoldings((prev) => prev.filter((h) => h.ticker !== ticker));
    } catch (err: any) {
      alert(`Failed to remove ${ticker}: ${err?.message}`);
    }
  };

  const handleEdit = (holding: PortfolioHolding) => {
    setEditHolding(holding);
    setIsModalOpen(true);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadHoldings();
    setRefreshing(false);
  };

  // ── Summary calculations ──────────────────────────────────────────────────

  const readyHoldings = holdings.filter((h) => h.current_value !== null);

  const totalInvested = holdings.reduce((sum, h) => sum + h.cost_basis, 0);
  const totalCurrentValue = readyHoldings.reduce(
    (sum, h) => sum + (h.current_value ?? 0),
    0,
  );
  const totalPnL = totalCurrentValue - totalInvested;
  const totalPnLPct = totalInvested > 0 ? (totalPnL / totalInvested) * 100 : null;
  const overallIsUp = totalPnL >= 0;

  const buySCount = holdings.filter((h) => h.current_signal === "buy").length;
  const sellCount = holdings.filter((h) => h.current_signal === "sell").length;
  const holdCount = holdings.filter((h) => h.current_signal === "hold").length;

  const fmt = (v: number) =>
    `₹${Math.abs(v).toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;

  return (
    <>
      <Head>
        <title>My Portfolio – NSE Simulator</title>
        <meta name="description" content="Track your NSE stock portfolio with ML signals" />
      </Head>

      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
                title="Back to Simulator"
              >
                <ArrowLeft size={16} />
              </Link>
              <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center">
                <Briefcase size={16} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">My Portfolio</h1>
                <p className="text-xs text-slate-500">Track your holdings with ML signals</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="btn-secondary flex items-center gap-2 text-sm"
                title="Refresh"
              >
                <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
                Refresh
              </button>
              <button
                onClick={() => {
                  setEditHolding(null);
                  setIsModalOpen(true);
                }}
                className="btn-primary flex items-center gap-2 text-sm"
              >
                <Plus size={14} />
                Add Holding
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Error */}
          {error && (
            <div className="mb-6 bg-red-900/20 border border-red-700/40 rounded-xl p-4 text-red-400 text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="card animate-pulse">
                  <div className="h-3 bg-slate-700 rounded w-20 mb-2" />
                  <div className="h-6 bg-slate-700 rounded w-28" />
                </div>
              ))}
            </div>
          )}

          {/* Summary cards — only show when we have data */}
          {!loading && holdings.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
              <SummaryCard
                label="Total Invested"
                value={fmt(totalInvested)}
                sub={`${holdings.length} holding${holdings.length !== 1 ? "s" : ""}`}
                color="neutral"
              />
              <SummaryCard
                label="Current Value"
                value={readyHoldings.length < holdings.length ? "Calculating…" : fmt(totalCurrentValue)}
                sub={readyHoldings.length < holdings.length ? `${holdings.length - readyHoldings.length} pending` : undefined}
                color="neutral"
              />
              <SummaryCard
                label="Total P&amp;L"
                value={
                  readyHoldings.length < holdings.length
                    ? "—"
                    : `${overallIsUp ? "+" : "−"}${fmt(totalPnL)}`
                }
                sub={
                  totalPnLPct !== null && readyHoldings.length === holdings.length
                    ? `${overallIsUp ? "+" : "−"}${Math.abs(totalPnLPct).toFixed(2)}%`
                    : undefined
                }
                color={
                  readyHoldings.length < holdings.length
                    ? "neutral"
                    : overallIsUp
                    ? "green"
                    : "red"
                }
              />
              <SummaryCard
                label="Signals"
                value={`${buySCount}B · ${holdCount}H · ${sellCount}S`}
                sub="buy · hold · sell"
                color={buySCount > sellCount ? "green" : sellCount > buySCount ? "red" : "neutral"}
              />
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && holdings.length === 0 && (
            <div className="text-center py-24">
              <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Briefcase size={32} className="text-slate-600" />
              </div>
              <h2 className="text-lg font-semibold text-white mb-2">No holdings yet</h2>
              <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">
                Add a stock you own to track its current value, P&amp;L, and model signals. The
                stock must already be in the{" "}
                <Link href="/" className="text-blue-400 hover:underline">
                  simulator
                </Link>
                .
              </p>
              <button
                onClick={() => {
                  setEditHolding(null);
                  setIsModalOpen(true);
                }}
                className="btn-primary inline-flex items-center gap-2"
              >
                <Plus size={16} />
                Add your first holding
              </button>
            </div>
          )}

          {/* Holdings grid */}
          {!loading && holdings.length > 0 && (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-medium text-slate-400">
                  {holdings.length} holding{holdings.length !== 1 ? "s" : ""}
                  {" · "}
                  <span className="text-slate-500 text-xs">
                    Click any card to see the full chart with your entry price
                  </span>
                </h2>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {holdings.map((h) => (
                  <PortfolioHoldingCard
                    key={h.ticker}
                    holding={h}
                    onEdit={handleEdit}
                    onRemove={handleRemove}
                  />
                ))}
              </div>
            </>
          )}
        </main>
      </div>

      <AddHoldingModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setEditHolding(null);
        }}
        onSaved={loadHoldings}
        editHolding={editHolding}
      />
    </>
  );
}
