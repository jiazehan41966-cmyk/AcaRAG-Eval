"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface EvalRun {
  run_id: string;
  run_name: string;
  status: string;
  cases: number;
  created_at: string;
  metrics: {
    faithfulness: number;
    answer_relevancy: number;
    question_coverage: number;
    context_recall: number;
    citation_accuracy: number;
  };
  failure_summary: {
    failure_rate: number;
    retrieval_miss?: number;
    rerank_error?: number;
    citation_error?: number;
    unsupported_claim?: number;
    incomplete_answer?: number;
  };
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    try {
      const res = await fetch("/api/eval/runs");
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
      }
    } catch (error) {
      console.error("Failed to fetch eval runs", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Evaluation Runs</h1>
        <p className="subtitle">View historical RAGAS and DeepEval results</p>
      </div>

      <div className="card">
        {loading ? (
          <div>Loading runs...</div>
        ) : runs.length === 0 ? (
          <div className="empty-state">No evaluation runs found.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Run Name</th>
                <th>Status</th>
                <th>Cases</th>
                <th>Faithfulness</th>
                <th>Context Recall</th>
                <th>Failure Rate</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td>
                    <strong>{run.run_name}</strong>
                    <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      {run.run_id.substring(0, 8)}
                    </div>
                  </td>
                  <td>
                    <span style={{ 
                      padding: "0.2rem 0.5rem", 
                      borderRadius: "1rem", 
                      fontSize: "0.85rem",
                      backgroundColor: run.status === "completed" ? "#e6f4ea" : "#fef7e0",
                      color: run.status === "completed" ? "#1e8e3e" : "#b06000"
                    }}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.cases}</td>
                  <td>{(run.metrics.faithfulness * 100).toFixed(1)}%</td>
                  <td>{(run.metrics.context_recall * 100).toFixed(1)}%</td>
                  <td>{(run.failure_summary.failure_rate * 100).toFixed(1)}%</td>
                  <td>{new Date(run.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
