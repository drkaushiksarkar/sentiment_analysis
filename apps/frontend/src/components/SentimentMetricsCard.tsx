import type { SentimentMetrics } from "../api/metrics";

type Props = {
  metrics: SentimentMetrics | null;
  isLoading: boolean;
  error?: string | null;
};

const LABEL_COLORS: Record<string, string> = {
  positive: "#16a34a",
  neutral: "#0f172a",
  negative: "#dc2626",
};

export function SentimentMetricsCard({ metrics, isLoading, error }: Props) {
  const timelinePoints = metrics?.timeline ?? [];
  const confidences = timelinePoints.map((point) => point.confidence);
  const maxConfidence = Math.max(0.01, ...confidences);
  const sparklinePoints = timelinePoints
    .map((point, idx) => {
      const x = (idx / Math.max(1, timelinePoints.length - 1)) * 100;
      const y = 40 - (point.confidence / maxConfidence) * 40;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <section className="card card--wide metrics-card">
      <header className="card__header">
        <div>
          <p className="card__label">Insights</p>
          <h3>Sentiment performance</h3>
        </div>
      </header>
      <div className="card__body">
        {isLoading && <p className="muted">Loading metrics…</p>}
        {error && <p className="error">{error}</p>}
        {!isLoading && !error && metrics ? (
          <div className="metrics-grid">
            <div>
              <p className="muted">Total requests</p>
              <p className="summary-value">{metrics.total_requests}</p>
            </div>
            <div>
              <p className="muted">Avg. confidence</p>
              <p className="summary-value">{Math.round(metrics.average_confidence * 100)}%</p>
            </div>
            <div className="label-breakdown">
              <p className="muted">Label distribution</p>
              <ul>
                {Object.entries(metrics.label_counts).map(([label, count]) => (
                  <li key={label}>
                    <span
                      className="label-pill"
                      style={{ backgroundColor: LABEL_COLORS[label], color: "#fff" }}
                    >
                      {label}
                    </span>
                    <span>{count}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="sparkline">
              <p className="muted">Recent confidence</p>
              {timelinePoints.length ? (
                <svg viewBox="0 0 100 40" preserveAspectRatio="none">
                  <polyline points={sparklinePoints} fill="none" stroke="#2563eb" strokeWidth="2" />
                </svg>
              ) : (
                <p className="muted">No data</p>
              )}
            </div>
            <div className="recent-list">
              <p className="muted">Latest predictions</p>
              <ul>
                {metrics.recent_predictions.slice(-5).reverse().map((entry) => (
                  <li key={entry.timestamp}>
                    <span>{entry.label.toUpperCase()}</span>
                    <span>{Math.round(entry.confidence * 100)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
