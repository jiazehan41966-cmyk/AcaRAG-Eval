"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type EvalRun = {
  run_id: string;
  run_name: string;
  cases: number;
  status: string;
  metrics?: Record<string, number>;
  channels?: Record<string, unknown>;
  failure_summary?: Record<string, number>;
  created_at: string;
};

type DocumentRow = {
  id: string;
  filename: string;
  parse_status: string;
  metadata?: Record<string, unknown>;
};

async function getJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return fallback;
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const latest = runs[0];

  async function refresh() {
    const [runRows, docRows] = await Promise.all([
      getJson<EvalRun[]>("/api/eval/runs", []),
      getJson<DocumentRow[]>("/api/documents", [])
    ]);
    setRuns(runRows);
    setDocuments(docRows);
  }

  useEffect(() => {
    void refresh();
  }, []);

  const failureRate = useMemo(() => {
    const rate = latest?.failure_summary?.failure_rate;
    return typeof rate === "number" ? `${(rate * 100).toFixed(1)}%` : "n/a";
  }, [latest]);

  return (
    <div className="stack">
      <section className="hero">
        <div className="panel">
          <p className="muted">Academic Agentic RAG Evaluation Platform</p>
          <h1>Traceable paper QA, measured end to end.</h1>
          <p className="muted">
            Monitor document ingestion, retrieval quality, RAGAS/DeepEval channels, failure attribution,
            and trace coverage from one lightweight React dashboard.
          </p>
          <div className="toolbar">
            <button onClick={() => void refresh()}>Refresh</button>
            <a href={`${API_BASE}/ui/documents`}>Classic Documents UI</a>
            <a href={`${API_BASE}/ui/chat`}>Classic Chat UI</a>
          </div>
        </div>
        <div className="panel stack">
          <div className="metric">
            <span className="muted">Documents</span>
            <strong>{documents.length}</strong>
          </div>
          <div className="metric">
            <span className="muted">Eval runs</span>
            <strong>{runs.length}</strong>
          </div>
          <div className="metric">
            <span className="muted">Latest failure rate</span>
            <strong>{failureRate}</strong>
          </div>
        </div>
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Latest Metrics</h2>
          <p>Faithfulness: {latest?.metrics?.faithfulness?.toFixed?.(4) ?? "n/a"}</p>
          <p>Answer relevancy: {latest?.metrics?.answer_relevancy?.toFixed?.(4) ?? "n/a"}</p>
          <p>Context recall: {latest?.metrics?.context_recall?.toFixed?.(4) ?? "n/a"}</p>
          <p>Citation accuracy: {latest?.metrics?.citation_accuracy?.toFixed?.(4) ?? "n/a"}</p>
        </div>
        <div className="panel">
          <h2>Recent Documents</h2>
          <div className="stack">
            {documents.slice(0, 5).map((doc) => (
              <div className="case" key={doc.id}>
                <strong>{String(doc.metadata?.title || doc.filename)}</strong>
                <p className="muted">{doc.parse_status}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="panel">
          <h2>Recent Runs</h2>
          <div className="stack">
            {runs.slice(0, 5).map((run) => (
              <div className="case" key={run.run_id}>
                <strong>{run.run_name}</strong>
                <p className="muted">
                  {run.status} / {run.cases} cases
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
