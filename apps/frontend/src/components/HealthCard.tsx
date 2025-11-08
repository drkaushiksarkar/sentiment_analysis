import type { HealthResponse } from "../api/health";

type HealthCardProps = {
  title: string;
  data: HealthResponse | null;
  isLoading: boolean;
  error?: string | null;
};

const statusColor: Record<string, string> = {
  ok: "status--ok",
  ready: "status--ok",
  healthy: "status--ok",
  degraded: "status--warn",
};

export function HealthCard({ title, data, isLoading, error }: HealthCardProps) {
  const status = data?.status?.toLowerCase() ?? null;
  const badgeClass = status ? statusColor[status] ?? "status--warn" : "";

  return (
    <article className="card">
      <header className="card__header">
        <div>
          <p className="card__label">Service</p>
          <h3>{title}</h3>
        </div>
        {status && !isLoading && !error ? (
          <span className={`status-badge ${badgeClass}`}>{data?.status}</span>
        ) : null}
      </header>
      <div className="card__body">
        {isLoading && <p className="muted">Checking...</p>}
        {error && <p className="error">{error}</p>}
        {!isLoading && !error && data ? (
          <ul className="card__list">
            {data.service && (
              <li>
                <span className="muted">Service:</span> {data.service}
              </li>
            )}
            {data.environment && (
              <li>
                <span className="muted">Environment:</span> {data.environment}
              </li>
            )}
            {data.detail && (
              <li>
                <span className="muted">Detail:</span> {data.detail}
              </li>
            )}
          </ul>
        ) : null}
      </div>
    </article>
  );
}
