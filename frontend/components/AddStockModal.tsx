import { useState, useRef, useEffect } from "react";
import { X, Search, AlertCircle, CheckCircle, Loader2 } from "lucide-react";
import { addStock } from "../lib/api";

interface AddStockModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdded: () => void;
}

const EXAMPLE_TICKERS = [
  "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
  "WIPRO", "SBIN", "BAJFINANCE", "HINDUNILVR", "ITC",
];

type Status = "idle" | "loading" | "success" | "error";

export default function AddStockModal({ isOpen, onClose, onAdded }: AddStockModalProps) {
  const [ticker, setTicker] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTicker("");
      setStatus("idle");
      setMessage("");
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const clean = ticker.trim().toUpperCase().replace(/\.NS$/, "").replace(/\.BO$/, "");
    if (!clean) return;

    setStatus("loading");
    setMessage("");

    try {
      const result = await addStock(clean);
      setStatus("success");
      setMessage(`${result.display_name} added! ${result.message}`);
      onAdded();
      setTimeout(() => {
        onClose();
      }, 2500);
    } catch (err: any) {
      setStatus("error");
      const detail = err?.response?.data?.detail || err?.message || "Unknown error";
      setMessage(detail);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-white">Add NSE Stock</h2>
            <p className="text-sm text-slate-400 mt-0.5">Enter an NSE ticker symbol to start tracking</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              ref={inputRef}
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. RELIANCE, TCS, INFY"
              className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-9 pr-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 font-mono text-sm transition-colors"
              disabled={status === "loading" || status === "success"}
              maxLength={20}
            />
          </div>

          {/* Warning */}
          <div className="bg-amber-900/20 border border-amber-700/40 rounded-lg p-3 text-xs text-amber-300 flex items-start gap-2">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <span>
              <strong>First run takes 2–3 minutes.</strong> The system will fetch 2 years of price data,
              scrape news, score sentiment with FinBERT, and train an XGBoost model.
            </span>
          </div>

          {/* Status message */}
          {status === "success" && (
            <div className="flex items-start gap-2 text-sm text-green-400 bg-green-900/20 border border-green-700/40 rounded-lg p-3">
              <CheckCircle size={14} className="shrink-0 mt-0.5" />
              <span>{message}</span>
            </div>
          )}
          {status === "error" && (
            <div className="flex items-start gap-2 text-sm text-red-400 bg-red-900/20 border border-red-700/40 rounded-lg p-3">
              <AlertCircle size={14} className="shrink-0 mt-0.5" />
              <span>{message}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={!ticker.trim() || status === "loading" || status === "success"}
            className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed py-3"
          >
            {status === "loading" ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Starting pipeline…
              </>
            ) : status === "success" ? (
              <>
                <CheckCircle size={16} />
                Added!
              </>
            ) : (
              "Add Stock"
            )}
          </button>
        </form>

        {/* Example tickers */}
        <div className="mt-5">
          <p className="text-xs text-slate-500 mb-2">Popular NSE stocks:</p>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLE_TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => setTicker(t)}
                className="text-xs font-mono bg-slate-700 hover:bg-slate-600 text-slate-300 px-2 py-1 rounded transition-colors disabled:opacity-50"
                disabled={status === "loading" || status === "success"}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
