export type HealthResponse = {
  status: string;
  service?: string;
  environment?: string;
  detail?: string;
};

export async function fetchHealth(path: "live" | "ready"): Promise<HealthResponse> {
  const response = await fetch(`/api/health/${path}`);
  if (!response.ok) {
    throw new Error(`Health check ${path} failed (${response.status})`);
  }
  return (await response.json()) as HealthResponse;
}
