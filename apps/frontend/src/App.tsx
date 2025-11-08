import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service?: string;
  environment?: string;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/health/live")
      .then((res) => res.json())
      .then((data) => setHealth(data as HealthResponse))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <main>
      <header>
        <h1>Sentiment Intelligence Platform</h1>
        <p>React + FastAPI scaffolding ready for further development.</p>
      </header>
      <section>
        <h2>Backend Health</h2>
        {health ? (
          <code>{JSON.stringify(health, null, 2)}</code>
        ) : error ? (
          <p className="error">{error}</p>
        ) : (
          <p>Loading...</p>
        )}
      </section>
    </main>
  );
}

export default App;
