import { FormEvent, useState } from "react";
import { analyzeSentiment, type SentimentResult } from "../api/sentiment";

export function SentimentPlayground() {
  const [text, setText] = useState("I love how seamless this experience is!");
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!text.trim()) {
      setError("Enter some text to analyze");
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const response = await analyzeSentiment(text);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze sentiment");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="card card--wide">
      <header className="card__header">
        <div>
          <p className="card__label">Playground</p>
          <h3>Try the sentiment endpoint</h3>
        </div>
      </header>
      <div className="card__body">
        <form className="playground" onSubmit={handleSubmit}>
          <label htmlFor="text" className="muted">
            Text to analyze
          </label>
          <textarea
            id="text"
            rows={4}
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste customer feedback, reviews, or release notes..."
          />
          <button className="primary" type="submit" disabled={isLoading}>
            {isLoading ? "Scoring..." : "Analyze sentiment"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
        {result && !error ? (
          <div className="playground__result">
            <p className="summary-value">{result.label.toUpperCase()}</p>
            <dl>
              <div>
                <dt>Score</dt>
                <dd>{result.score}</dd>
              </div>
              <div>
                <dt>Confidence</dt>
                <dd>{Math.round(result.confidence * 100)}%</dd>
              </div>
              <div>
                <dt>Tokens analyzed</dt>
                <dd>{result.tokens_analyzed}</dd>
              </div>
            </dl>
          </div>
        ) : null}
      </div>
    </section>
  );
}
