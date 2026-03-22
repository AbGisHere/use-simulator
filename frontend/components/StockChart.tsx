import { useState, useMemo } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  ReferenceLine,
} from "recharts";
import { subMonths, subYears, parseISO, isAfter } from "date-fns";
import { ChartDataPoint, FutureDataPoint } from "../lib/api";

// ── Combined data structure for the chart ─────────────────────────────────────
// Historical and future points are merged into one array so Recharts renders
// them on a single continuous X-axis.
interface CombinedPoint {
  date: string;
  isFuture: boolean;

  // Historical fields
  actualPrice?: number;
  updatedPredLine?: number;   // latest model's predicted path (price-offset)
  initialPredLine?: number;   // first-run model's predicted path (price-offset)
  upperBand?: number;
  lowerBand?: number;
  signal?: string;
  sentimentScore?: number;
  confidence?: number;

  // Future fields
  projectedPrice?: number;
  futureUpper?: number;
  futureLower?: number;
}

interface StockChartProps {
  data: ChartDataPoint[];
  futureData: FutureDataPoint[];
  ticker: string;
  buyPrice?: number;   // optional — shows a horizontal entry price line
  buyQuantity?: number; // optional — used to show unrealised P&L in tooltip
}

type DateRange = "1M" | "3M" | "6M" | "1Y" | "ALL";
const DATE_RANGE_OPTIONS: DateRange[] = ["1M", "3M", "6M", "1Y", "ALL"];

// Convert a direction (1/0) + base price into a price-offset line value
// so predicted direction shows up as a visible line on the price chart.
function directionToPrice(direction: number | null | undefined, basePrice: number): number | undefined {
  if (direction == null || basePrice == null) return undefined;
  return basePrice * (1 + (direction === 1 ? 0.006 : -0.006));
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const pt: CombinedPoint = payload[0]?.payload ?? {};

  const fmt = (v: number) =>
    `₹${v.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div style={{
      background: "#0f172a",
      border: "1px solid #334155",
      borderRadius: 8,
      padding: "10px 14px",
      fontSize: 12,
      minWidth: 180,
    }}>
      <p style={{ color: "#64748b", marginBottom: 6 }}>{label}</p>

      {pt.isFuture ? (
        <>
          <p style={{ color: "#22c55e" }}>
            Projected: {pt.projectedPrice != null ? fmt(pt.projectedPrice) : "—"}
          </p>
          {pt.futureUpper != null && pt.futureLower != null && (
            <p style={{ color: "#64748b" }}>
              Range: {fmt(pt.futureLower)} – {fmt(pt.futureUpper)}
            </p>
          )}
          {pt.confidence != null && (
            <p style={{ color: "#94a3b8" }}>
              Confidence: {(pt.confidence * 100).toFixed(1)}%
            </p>
          )}
        </>
      ) : (
        <>
          {pt.actualPrice != null && (
            <p style={{ color: "#3b82f6" }}>Actual: {fmt(pt.actualPrice)}</p>
          )}
          {pt.updatedPredLine != null && (
            <p style={{ color: "#f97316" }}>
              Latest prediction: {pt.updatedPredLine > (pt.actualPrice ?? 0) ? "↑ Up" : "↓ Down"}
              {pt.confidence != null && (
                <span style={{ color: "#64748b" }}> ({(pt.confidence * 100).toFixed(0)}%)</span>
              )}
            </p>
          )}
          {pt.initialPredLine != null && (
            <p style={{ color: "#94a3b8" }}>
              Initial prediction: {pt.initialPredLine > (pt.actualPrice ?? 0) ? "↑ Up" : "↓ Down"}
            </p>
          )}
          {pt.signal && pt.signal !== "hold" && (
            <p style={{ color: pt.signal === "buy" ? "#22c55e" : "#ef4444", fontWeight: 700 }}>
              Signal: {pt.signal.toUpperCase()}
            </p>
          )}
          {pt.sentimentScore != null && pt.sentimentScore !== 0 && (
            <p style={{ color: pt.sentimentScore > 0 ? "#22c55e" : "#ef4444" }}>
              Sentiment: {pt.sentimentScore > 0 ? "+" : ""}{pt.sentimentScore.toFixed(3)}
            </p>
          )}
        </>
      )}
    </div>
  );
}

// ── Toggle legend item ────────────────────────────────────────────────────────
function ToggleItem({
  checked, onChange, color, label, dashed,
}: {
  checked: boolean; onChange: () => void; color: string; label: string; dashed?: boolean;
}) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 12 }}>
      <span style={{
        display: "inline-block",
        width: 24, height: 2,
        background: checked ? color : "#334155",
        borderRadius: 1,
        borderTop: dashed ? `2px dashed ${checked ? color : "#334155"}` : undefined,
        transition: "background 0.15s",
      }} />
      <input type="checkbox" checked={checked} onChange={onChange} style={{ display: "none" }} />
      <span style={{ color: checked ? "#cbd5e1" : "#475569" }}>{label}</span>
    </label>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function StockChart({ data, futureData, ticker, buyPrice, buyQuantity }: StockChartProps) {
  const [range, setRange] = useState<DateRange>("1Y");
  const [showActual, setShowActual] = useState(true);
  const [showUpdated, setShowUpdated] = useState(true);
  const [showInitial, setShowInitial] = useState(true);
  const [showFuture, setShowFuture] = useState(true);
  const [showBands, setShowBands] = useState(true);
  const [showBuy, setShowBuy] = useState(true);
  const [showSell, setShowSell] = useState(true);
  const [showSentiment, setShowSentiment] = useState(true);

  // ── Build combined dataset ─────────────────────────────────────────────────
  const combined = useMemo<CombinedPoint[]>(() => {
    const cutoff: Record<DateRange, Date> = {
      "1M": subMonths(new Date(), 1),
      "3M": subMonths(new Date(), 3),
      "6M": subMonths(new Date(), 6),
      "1Y": subYears(new Date(), 1),
      ALL: new Date(0),
    };
    const threshold = cutoff[range];

    const historical: CombinedPoint[] = data
      .filter((d) => isAfter(parseISO(d.date), threshold))
      .map((d) => ({
        date: d.date,
        isFuture: false,
        actualPrice: d.actual_price,
        updatedPredLine: directionToPrice(d.predicted_direction, d.actual_price),
        initialPredLine: directionToPrice(d.initial_predicted_direction, d.actual_price),
        upperBand: d.upper_band ?? undefined,
        lowerBand: d.lower_band ?? undefined,
        signal: d.signal,
        sentimentScore: d.sentiment_score,
        confidence: d.confidence ?? undefined,
      }));

    // For future data, always show (no date filter — they're always in the future)
    const future: CombinedPoint[] = (futureData ?? []).map((d) => ({
      date: d.date,
      isFuture: true,
      projectedPrice: d.projected_price,
      futureUpper: d.upper_band,
      futureLower: d.lower_band,
      confidence: d.confidence,
    }));

    return [...historical, ...future];
  }, [data, futureData, range]);

  // "Today" divider: the first future date
  const todayDate = useMemo(
    () => futureData?.[0]?.date ?? null,
    [futureData]
  );

  // Price domain including both historical and future
  const priceMin = useMemo(() => {
    const vals = combined.flatMap((d) => [
      d.lowerBand, d.futureLower, d.actualPrice, d.projectedPrice,
    ].filter((v): v is number => v != null));
    return vals.length ? Math.min(...vals) * 0.97 : 0;
  }, [combined]);

  const priceMax = useMemo(() => {
    const vals = combined.flatMap((d) => [
      d.upperBand, d.futureUpper, d.actualPrice, d.projectedPrice,
    ].filter((v): v is number => v != null));
    return vals.length ? Math.max(...vals) * 1.03 : 100;
  }, [combined]);

  const formatPrice = (v: number) =>
    `₹${v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0)}`;

  const buyPoints = useMemo(() => combined.filter((d) => d.signal === "buy"), [combined]);
  const sellPoints = useMemo(() => combined.filter((d) => d.signal === "sell"), [combined]);

  if (combined.length === 0) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 320, color: "#64748b" }}>
        No data for selected range
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Controls row */}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 12 }}>

        {/* Date range buttons */}
        <div style={{ display: "flex", gap: 4, background: "#0f172a", borderRadius: 8, padding: 4 }}>
          {DATE_RANGE_OPTIONS.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              style={{
                padding: "4px 12px",
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 500,
                background: range === r ? "#2563eb" : "transparent",
                color: range === r ? "#fff" : "#64748b",
                transition: "all 0.15s",
              }}
            >
              {r}
            </button>
          ))}
        </div>

        {/* Toggle legend */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 14 }}>
          <ToggleItem checked={showActual} onChange={() => setShowActual(!showActual)} color="#3b82f6" label="Actual Price" />
          <ToggleItem checked={showUpdated} onChange={() => setShowUpdated(!showUpdated)} color="#f97316" label="Latest Prediction" dashed />
          <ToggleItem checked={showInitial} onChange={() => setShowInitial(!showInitial)} color="#94a3b8" label="Initial Prediction" dashed />
          <ToggleItem checked={showFuture} onChange={() => setShowFuture(!showFuture)} color="#22c55e" label="30-Day Forecast" dashed />
          <ToggleItem checked={showBands} onChange={() => setShowBands(!showBands)} color="#f9731640" label="Confidence Bands" />
          <ToggleItem checked={showBuy} onChange={() => setShowBuy(!showBuy)} color="#22c55e" label="Buy Signals" />
          <ToggleItem checked={showSell} onChange={() => setShowSell(!showSell)} color="#ef4444" label="Sell Signals" />
          <ToggleItem checked={showSentiment} onChange={() => setShowSentiment(!showSentiment)} color="#a855f7" label="Sentiment" />
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={combined} margin={{ top: 10, right: 60, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickLine={false}
            axisLine={{ stroke: "#334155" }}
            interval="preserveStartEnd"
            minTickGap={60}
          />
          <YAxis
            yAxisId="price"
            domain={[priceMin, priceMax]}
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatPrice}
            width={55}
          />
          <YAxis
            yAxisId="sentiment"
            orientation="right"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: "#64748b" }}
            tickLine={false}
            axisLine={false}
            width={30}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* "Today" vertical divider */}
          {todayDate && (
            <ReferenceLine
              yAxisId="price"
              x={todayDate}
              stroke="#334155"
              strokeDasharray="6 3"
              label={{ value: "Today", position: "insideTopRight", fill: "#475569", fontSize: 11 }}
            />
          )}

          {/* ── Historical confidence band ── */}
          {showBands && (
            <Area yAxisId="price" type="monotone" dataKey="upperBand"
              fill="#f97316" stroke="none" fillOpacity={0.07} connectNulls={false} />
          )}
          {showBands && (
            <Area yAxisId="price" type="monotone" dataKey="lowerBand"
              fill="#f97316" stroke="none" fillOpacity={0.07} connectNulls={false} />
          )}

          {/* ── Future confidence band ── */}
          {showBands && showFuture && (
            <Area yAxisId="price" type="monotone" dataKey="futureUpper"
              fill="#22c55e" stroke="none" fillOpacity={0.07} connectNulls={false} />
          )}
          {showBands && showFuture && (
            <Area yAxisId="price" type="monotone" dataKey="futureLower"
              fill="#22c55e" stroke="none" fillOpacity={0.07} connectNulls={false} />
          )}

          {/* ── LINE 1: Actual price (blue solid) ── */}
          {showActual && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="actualPrice"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#3b82f6" }}
              connectNulls={false}
              name="Actual Price"
            />
          )}

          {/* ── LINE 2: Initial prediction (gray dashed) ── */}
          {showInitial && (
            <Line
              yAxisId="price"
              type="stepAfter"
              dataKey="initialPredLine"
              stroke="#94a3b8"
              strokeWidth={1.5}
              strokeDasharray="3 4"
              dot={false}
              connectNulls={false}
              name="Initial Prediction"
            />
          )}

          {/* ── LINE 3: Latest/updated prediction (orange dashed) ── */}
          {showUpdated && (
            <Line
              yAxisId="price"
              type="stepAfter"
              dataKey="updatedPredLine"
              stroke="#f97316"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={false}
              connectNulls={false}
              name="Latest Prediction"
            />
          )}

          {/* ── LINE 4: Future projected price (green dotted) ── */}
          {showFuture && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="projectedPrice"
              stroke="#22c55e"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              connectNulls={false}
              name="30-Day Forecast"
            />
          )}

          {/* ── Sentiment line ── */}
          {showSentiment && (
            <Line
              yAxisId="sentiment"
              type="monotone"
              dataKey="sentimentScore"
              stroke="#a855f7"
              strokeWidth={1.5}
              dot={false}
              opacity={0.8}
              connectNulls={false}
              name="Sentiment"
            />
          )}

          {/* ── Buy signals ── */}
          {showBuy && buyPoints.map((d, i) => (
            <ReferenceDot
              key={`buy-${i}`}
              yAxisId="price"
              x={d.date}
              y={d.actualPrice}
              r={6}
              fill="#22c55e"
              stroke="#16a34a"
              strokeWidth={1.5}
              label=""
            />
          ))}

          {/* ── Sell signals ── */}
          {showSell && sellPoints.map((d, i) => (
            <ReferenceDot
              key={`sell-${i}`}
              yAxisId="price"
              x={d.date}
              y={d.actualPrice}
              r={6}
              fill="#ef4444"
              stroke="#dc2626"
              strokeWidth={1.5}
              label=""
            />
          ))}

          {/* ── Buy price entry line (portfolio mode) ── */}
          {buyPrice != null && (
            <ReferenceLine
              yAxisId="price"
              y={buyPrice}
              stroke="#facc15"
              strokeWidth={1.5}
              strokeDasharray="6 3"
              label={{
                value: `Entry ₹${buyPrice.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`,
                position: "insideTopLeft",
                fill: "#facc15",
                fontSize: 11,
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend footer */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, fontSize: 11, color: "#475569" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 20, height: 2, background: "#3b82f6" }} />
          Actual closing price
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 20, height: 0, borderTop: "2px dashed #94a3b8" }} />
          Initial model prediction (1st run)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 20, height: 0, borderTop: "2px dashed #f97316" }} />
          Latest model prediction (after refresh)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 20, height: 0, borderTop: "2px dashed #22c55e" }} />
          30-day forecast (projected)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#22c55e" }} />
          Buy signal (&gt;60% confidence)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#ef4444" }} />
          Sell signal (&gt;60% confidence)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", width: 20, height: 0, borderTop: "2px solid #a855f7" }} />
          Sentiment score
        </span>
      </div>
    </div>
  );
}
