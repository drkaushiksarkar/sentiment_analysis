export type PredictionSummary = {
  label: "positive" | "negative" | "neutral";
  confidence: number;
  timestamp: string;
};

export type TimelinePoint = {
  timestamp: string;
  confidence: number;
};

export type SentimentMetrics = {
  total_requests: number;
  label_counts: Record<"positive" | "negative" | "neutral", number>;
  average_confidence: number;
  recent_predictions: PredictionSummary[];
  timeline: TimelinePoint[];
};

export async function fetchSentimentMetrics(): Promise<SentimentMetrics> {
  const response = await fetch("/api/v1/metrics/sentiment");
  if (!response.ok) {
    throw new Error("Unable to load sentiment metrics");
  }
  return (await response.json()) as SentimentMetrics;
}
