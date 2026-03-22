import { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Cell,
} from "recharts";
import { ChartDataPoint, NewsItem } from "../lib/api";
import { format, parseISO } from "date-fns";

interface SentimentOverlayProps {
  data: ChartDataPoint[];
  news: NewsItem[];
}

interface TooltipPayload {
  date: string;
  sentiment_score: number;
}

function CustomTooltip({ active, payload, label, news, onDateClick }: any) {
  if (!active || !payload?.length) return null;

  const sentiment = payload[0]?.value as number;
  const dateNews = news.filter((n: NewsItem) => {
    if (!n.published_at) return false;
    try {
      return format(parseISO(n.published_at), "yyyy-MM-dd") === label;
    } catch {
      return false;
    }
  });

  return (
    <div className="custom-tooltip max-w-xs">
      <p className="font-medium text-slate-300 mb-1">{label}</p>
      <p className={`text-sm font-bold ${sentiment > 0 ? "text-green-400" : sentiment < 0 ? "text-red-400" : "text-slate-400"}`}>
        Sentiment: {sentiment > 0 ? "+" : ""}{sentiment.toFixed(3)}
      </p>
      {dateNews.length > 0 && (
        <div className="mt-2 space-y-1 border-t border-slate-600 pt-2">
          <p className="text-xs text-slate-500">{dateNews.length} article{dateNews.length !== 1 ? "s" : ""}:</p>
          {dateNews.slice(0, 3).map((n: NewsItem, i: number) => (
            <p key={i} className="text-xs text-slate-300 line-clamp-2">{n.headline}</p>
          ))}
          {dateNews.length > 3 && (
            <p className="text-xs text-slate-500">+{dateNews.length - 3} more</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function SentimentOverlay({ data, news }: SentimentOverlayProps) {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  // Filter to only days with non-zero sentiment
  const sentimentData = data
    .filter((d) => d.sentiment_score !== 0)
    .map((d) => ({
      date: d.date,
      sentiment_score: d.sentiment_score,
    }));

  const selectedNews = selectedDate
    ? news.filter((n) => {
        if (!n.published_at) return false;
        try {
          return format(parseISO(n.published_at), "yyyy-MM-dd") === selectedDate;
        } catch {
          return false;
        }
      })
    : [];

  const handleBarClick = (data: any) => {
    if (data?.activePayload?.[0]?.payload?.date) {
      setSelectedDate(data.activePayload[0].payload.date);
    }
  };

  if (sentimentData.length === 0) {
    return (
      <div className="flex items-center justify-center h-24 text-sm text-slate-500">
        No sentiment data available
      </div>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={sentimentData} onClick={handleBarClick} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
          <XAxis dataKey="date" hide />
          <YAxis domain={[-1, 1]} hide />
          <ReferenceLine y={0} stroke="#475569" strokeWidth={1} />
          <Tooltip
            content={(props) => (
              <CustomTooltip {...props} news={news} onDateClick={setSelectedDate} />
            )}
          />
          <Bar dataKey="sentiment_score" radius={[2, 2, 0, 0]}>
            {sentimentData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.sentiment_score > 0 ? "#22c55e" : "#ef4444"}
                opacity={selectedDate === entry.date ? 1 : 0.7}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Selected date news panel */}
      {selectedDate && selectedNews.length > 0 && (
        <div className="mt-3 p-3 bg-slate-900 rounded-lg border border-slate-700">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium text-slate-400">
              News for {selectedDate} ({selectedNews.length} article{selectedNews.length !== 1 ? "s" : ""})
            </p>
            <button
              onClick={() => setSelectedDate(null)}
              className="text-xs text-slate-500 hover:text-white"
            >
              ✕
            </button>
          </div>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {selectedNews.map((n, i) => (
              <div key={i} className="text-xs border-l-2 pl-2" style={{
                borderColor: n.sentiment_label === "positive" ? "#22c55e" :
                             n.sentiment_label === "negative" ? "#ef4444" : "#64748b"
              }}>
                <p className="text-slate-200 line-clamp-2">{n.headline}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-slate-500">{n.source}</span>
                  {n.sentiment_label && (
                    <span className={
                      n.sentiment_label === "positive" ? "text-green-400" :
                      n.sentiment_label === "negative" ? "text-red-400" : "text-slate-500"
                    }>
                      {n.sentiment_label}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
