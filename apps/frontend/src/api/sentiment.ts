export type SentimentPayload = {
  text: string;
};

export type SentimentResult = {
  label: string;
  score: number;
  confidence: number;
  tokens_analyzed: number;
};

export async function analyzeSentiment(text: string): Promise<SentimentResult> {
  const response = await fetch("/api/v1/sentiment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text } satisfies SentimentPayload),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Unable to analyze sentiment");
  }
  return (await response.json()) as SentimentResult;
}
