import { useRouter } from "next/router";
import { TrendingUp, TrendingDown, Minus, Edit2, Trash2 } from "lucide-react";
import { PortfolioHolding } from "../lib/api";

interface PortfolioHoldingCardProps {
  holding: PortfolioHolding;
  onEdit: (holding: PortfolioHolding) => void;
  onRemove: (ticker: string) => void;
}

function SignalBadge({ signal, confidence }: { signal: string; confidence: number | null }) {
  if (signal === "buy") {
    return (
      <span className="badge-green flex items-center gap-1">
        <TrendingUp size={10} />
        BUY {confidence ? `${(confidence * 100).toFixed(0)}%` : ""}
      </span>
    );
  }
  if (signal === "sell") {
    return (
      <span className="badge-red flex items-center gap-1">
        <TrendingDown size={10} />
        SELL {confidence ? `${(confidence * 100).toFixed(0)}%` : ""}
      </span>
    );
  }
  return (
    <span className="badge-yellow flex items-center gap-1">
      <Minus size={10} />
      HOLD
    </span>
  );
}

export default function PortfolioHoldingCard({
  holding,
  onEdit,
  onRemove,
}: PortfolioHoldingCardProps) {
  const router = useRouter();

  const pnlAbs = holding.pnl_abs;
  const pnlPct = holding.pnl_pct;
  const isUp = pnlAbs !== null && pnlAbs >= 0;
  const isProcessing = holding.current_price === null;

  const fmt = (v: number | null, prefix = "₹") =>
    v !== null
      ? `${prefix}${Math.abs(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
      : "—";

  return (
    <div
      className={`card cursor-pointer hover:border-slate-500 transition-all duration-150 group border-l-4 ${
        isProcessing ? "border-l-slate-600" : isUp ? "border-l-green-500" : "border-l-red-500"
      }`}
      onClick={() =>
        router.push(
          `/stock/${holding.ticker}?buyPrice=${holding.buy_price}&buyQuantity=${holding.quantity}`,
        )
      }
      role="button"
      tabIndex={0}
      onKeyDown={(e) =>
        e.key === "Enter" &&
        router.push(
          `/stock/${holding.ticker}?buyPrice=${holding.buy_price}&buyQuantity=${holding.quantity}`,
        )
      }
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-xs font-mono font-bold text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded">
            {holding.ticker}
          </span>
          <p className="text-xs text-slate-400 mt-1">
            {holding.quantity} shares @ ₹{holding.buy_price.toLocaleString("en-IN")}
          </p>
        </div>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            className="p-1 hover:bg-slate-700 rounded text-slate-500 hover:text-blue-400"
            onClick={(e) => {
              e.stopPropagation();
              onEdit(holding);
            }}
            title="Edit holding"
          >
            <Edit2 size={13} />
          </button>
          <button
            className="p-1 hover:bg-slate-700 rounded text-slate-500 hover:text-red-400"
            onClick={(e) => {
              e.stopPropagation();
              onRemove(holding.ticker);
            }}
            title="Remove holding"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {/* Current value */}
      {isProcessing ? (
        <div className="mb-3">
          <p className="text-slate-500 text-sm">Fetching data…</p>
        </div>
      ) : (
        <div className="mb-3">
          <p className="text-2xl font-bold text-white">
            {fmt(holding.current_value)}
          </p>
          <div className={`flex items-center gap-1.5 mt-0.5 text-sm font-medium ${isUp ? "text-green-400" : "text-red-400"}`}>
            {isUp ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
            <span>
              {isUp ? "+" : "−"}{fmt(pnlAbs)}{" "}
              <span className="text-xs font-normal opacity-80">
                ({isUp ? "+" : "−"}{pnlPct !== null ? Math.abs(pnlPct).toFixed(2) : "—"}%)
              </span>
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Cost ₹{holding.cost_basis.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between">
        <SignalBadge signal={holding.current_signal} confidence={holding.signal_confidence} />
        {holding.notes && (
          <span className="text-xs text-slate-500 truncate max-w-[120px]" title={holding.notes}>
            {holding.notes}
          </span>
        )}
      </div>
    </div>
  );
}
