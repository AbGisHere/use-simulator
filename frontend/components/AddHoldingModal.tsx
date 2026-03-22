import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { addHolding, updateHolding, PortfolioHolding } from "../lib/api";

interface AddHoldingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  editHolding?: PortfolioHolding | null;
}

export default function AddHoldingModal({
  isOpen,
  onClose,
  onSaved,
  editHolding,
}: AddHoldingModalProps) {
  const [ticker, setTicker] = useState("");
  const [buyPrice, setBuyPrice] = useState("");
  const [quantity, setQuantity] = useState("");
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isEdit = !!editHolding;

  useEffect(() => {
    if (editHolding) {
      setTicker(editHolding.ticker);
      setBuyPrice(String(editHolding.buy_price));
      setQuantity(String(editHolding.quantity));
      setNotes(editHolding.notes ?? "");
    } else {
      setTicker("");
      setBuyPrice("");
      setQuantity("");
      setNotes("");
    }
    setError(null);
  }, [editHolding, isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const price = parseFloat(buyPrice);
    const qty = parseFloat(quantity);

    if (!ticker || isNaN(price) || price <= 0 || isNaN(qty) || qty <= 0) {
      setError("Please fill in all fields with valid positive numbers.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const sym = ticker.toUpperCase().trim();
      if (isEdit) {
        await updateHolding(sym, price, qty, notes || undefined);
      } else {
        await addHolding(sym, price, qty, notes || undefined);
      }
      onSaved();
      onClose();
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to save holding.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-slate-800 border border-slate-700 rounded-2xl p-6 w-full max-w-md shadow-2xl">
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white"
        >
          <X size={16} />
        </button>

        <h2 className="text-lg font-semibold text-white mb-1">
          {isEdit ? "Edit Holding" : "Add Holding"}
        </h2>
        <p className="text-sm text-slate-400 mb-5">
          {isEdit
            ? "Update your buy price or quantity."
            : "Enter a stock you hold. The simulator must already be tracking it."}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Ticker */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              NSE Ticker
            </label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. RELIANCE, TCS, INFY"
              disabled={isEdit}
              className="input-field w-full disabled:opacity-50 disabled:cursor-not-allowed"
              autoFocus={!isEdit}
            />
          </div>

          {/* Buy Price */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Buy Price (₹ per share)
            </label>
            <input
              type="number"
              value={buyPrice}
              onChange={(e) => setBuyPrice(e.target.value)}
              placeholder="e.g. 2450.00"
              min="0.01"
              step="0.01"
              className="input-field w-full"
              autoFocus={isEdit}
            />
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Quantity (shares)
            </label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="e.g. 10"
              min="0.001"
              step="any"
              className="input-field w-full"
            />
          </div>

          {/* Notes */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Notes{" "}
              <span className="text-slate-500 font-normal">(optional)</span>
            </label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. Long-term hold, SIP purchase"
              className="input-field w-full"
            />
          </div>

          {error && (
            <div className="bg-red-900/20 border border-red-700/40 rounded-lg p-3 text-red-400 text-sm">
              {error}
            </div>
          )}

          <div className="flex gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary flex-1"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary flex-1 disabled:opacity-50"
            >
              {loading ? "Saving…" : isEdit ? "Save Changes" : "Add Holding"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
