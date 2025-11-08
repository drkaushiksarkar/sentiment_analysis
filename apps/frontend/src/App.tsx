import { useEffect, useMemo, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/health";
import { fetchSentimentMetrics, type SentimentMetrics } from "./api/metrics";
import { HealthCard } from "./components/HealthCard";
import { SentimentPlayground } from "./components/SentimentPlayground";
import { SentimentMetricsCard } from "./components/SentimentMetricsCard";

type HealthState = {
  live: HealthResponse | null;
  ready: HealthResponse | null;
};

function App() {
  const [health, setHealth] = useState<HealthState>({ live: null, ready: null });
  const [errors, setErrors] = useState<{ live: string | null; ready: string | null }>({
    live: null,
    ready: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [metrics, setMetrics] = useState<SentimentMetrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const refreshHealthAndMetrics = async () => {
    setIsLoading(true);
    setErrors({ live: null, ready: null });
    setMetricsError(null);
    try {
      const [live, ready, metricsResult] = await Promise.allSettled([
        fetchHealth("live"),
        fetchHealth("ready"),
        fetchSentimentMetrics(),
      ]);
      setHealth({
        live: live.status === "fulfilled" ? live.value : null,
        ready: ready.status === "fulfilled" ? ready.value : null,
      });
      setErrors({
        live: live.status === "rejected" ? live.reason?.message ?? "Unable to load" : null,
        ready: ready.status === "rejected" ? ready.reason?.message ?? "Unable to load" : null,
      });
      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value);
      } else {
        setMetrics(null);
        setMetricsError(metricsResult.reason?.message ?? "Unable to load metrics");
      }
      setLastChecked(new Date());
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void refreshHealthAndMetrics();
  }, []);

  const statusSummary = useMemo(() => {
    const statuses = [health.live?.status, health.ready?.status].filter(Boolean);
    if (!statuses.length) {
      return "Awaiting checks…";
    }
    const unique = new Set(statuses);
    return unique.size === 1 ? statuses[0] : "Mixed";
  }, [health]);

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Sentiment Intelligence Platform</p>
          <h1>Operations dashboard</h1>
          <p className="muted">
            Monitor backend services and keep an eye on core health checks before shipping new
            models or UI changes.
          </p>
        </div>
        <div className="hero__actions">
          <button onClick={() => refreshHealthAndMetrics()} disabled={isLoading} className="primary">
            {isLoading ? "Refreshing…" : "Refresh status"}
          </button>
          {lastChecked && <p className="muted">Last checked: {lastChecked.toLocaleTimeString()}</p>}
        </div>
      </header>

      <section className="grid">
        <HealthCard title="Live probe" data={health.live} isLoading={isLoading} error={errors.live} />
        <HealthCard
          title="Readiness probe"
          data={health.ready}
          isLoading={isLoading}
          error={errors.ready}
        />
        <article className="card card--wide">
          <header className="card__header">
            <div>
              <p className="card__label">Summary</p>
              <h3>Environment snapshot</h3>
            </div>
          </header>
          <div className="card__body">
            <p className="summary-value">{statusSummary}</p>
            <p className="muted">
              Live + readiness probes are polled via the FastAPI gateway (`/api/health/*`). Extend
              this section with latency charts or drift signals as the platform evolves.
            </p>
          </div>
        </article>
        <SentimentPlayground />
        <SentimentMetricsCard metrics={metrics} isLoading={isLoading} error={metricsError} />
      </section>
    </main>
  );
}

export default App;
