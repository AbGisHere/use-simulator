import { useRouter } from "next/router";
import { TrendingUp, TrendingDown, Clock, RefreshCw } from "lucide-react";
import { StockSummary } from "../lib/api";
import { formatDistanceToNow } from "date-fns";

interface StockCardProps {
  stock: StockSummary;
  onRemove: (ticker: string) => void;
}

function AccuracyBadge({ accuracy }: { accuracy: number }) {
  if (accuracy >= 55) {
    return <span className="badge-green">Accuracy {accuracy.toFixed(1)}%</span>;
  } else if (accuracy >= 50) {
    return <span className="badge-yellow">Accuracy {accuracy.toFixed(1)}%</span>;
  }
  return <span className="badge-red">Accuracy {accuracy.toFixed(1)}%</span>;
}

function borderColorClass(accuracy: number) {
  if (accuracy >= 55) return "border-l-4 border-l-green-500";
  if (accuracy >= 50) return "border-l-4 border-l-yellow-500";
  return "border-l-4 border-l-red-500";
}

export default function StockCard({ stock, onRemove }: StockCardProps) {
  const router = useRouter();

  const formattedPrice = stock.last_price
    ? `₹${stock.last_price.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    : "—";

  const lastUpdatedText = stock.last_updated
    ? formatDistanceToNow(new Date(stock.last_updated), { addSuffix: true })
    : "Processing…";

  return (
    <div
      className={`card cursor-pointer hover:border-slate-500 transition-all duration-150 group ${borderColorClass(stock.model_accuracy)}`}
      onClick={() => router.push(`/stock/${stock.ticker}`)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && router.push(`/stock/${stock.ticker}`)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-xs font-mono font-bold text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded">
            {stock.ticker}
          </span>
          <p className="text-sm text-slate-300 mt-1 line-clamp-1">{stock.display_name}</p>
        </div>
        <button
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-slate-700 rounded text-slate-500 hover:text-red-400"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(stock.ticker);
          }}
          title="Remove stock"
        >
          ×
        </button>
      </div>

      {/* Price */}
      <div className="mb-3">
        <p className="text-2xl font-bold text-white">{formattedPrice}</p>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between">
        <AccuracyBadge accuracy={stock.model_accuracy} />
        <div className="flex items-center gap-1 text-xs text-slate-500">
          <Clock size={11} />
          <span>{lastUpdatedText}</span>
        </div>
      </div>

      {/* Processing state */}
      {!stock.last_updated && (
        <div className="mt-2 flex items-center gap-2 text-xs text-yellow-400">
          <RefreshCw size={11} className="animate-spin" />
          <span>Fetching data & training model…</span>
        </div>
      )}
    </div>
  );
}
