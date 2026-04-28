"use client";

import { useEffect, useState } from "react";

interface TraceRecord {
  trace_id: string;
  name: string;
  timestamp: string;
  latency_ms: number;
  total_tokens: number;
  steps: any[];
}

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTraces();
  }, []);

  const fetchTraces = async () => {
    try {
      const res = await fetch("/api/traces");
      if (res.ok) {
        const data = await res.json();
        setTraces(data.traces || []);
      }
    } catch (error) {
      console.error("Failed to fetch traces", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Langfuse Traces</h1>
        <p className="subtitle">View agentic RAG workflow execution traces</p>
      </div>

      <div className="card">
        {loading ? (
          <div>Loading traces...</div>
        ) : traces.length === 0 ? (
          <div className="empty-state">No traces found. Run some chat queries or evals first.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Trace ID</th>
                <th>Name</th>
                <th>Steps</th>
                <th>Latency (ms)</th>
                <th>Tokens</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((trace) => (
                <tr key={trace.trace_id}>
                  <td>{trace.trace_id.substring(0, 8)}...</td>
                  <td><strong>{trace.name}</strong></td>
                  <td>{trace.steps?.length || 0}</td>
                  <td>{Math.round(trace.latency_ms)}</td>
                  <td>{trace.total_tokens}</td>
                  <td>{new Date(trace.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
