"use client";

import { useEffect, useState } from "react";

interface DocumentMetadata {
  doc_id: string;
  filename: string;
  upload_time: string;
  metadata?: {
    title?: string;
    authors?: string;
    year?: string;
    venue?: string;
  };
  chunk_count?: number;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/api/documents/");
      if (res.ok) {
        const data = await res.json();
        if (data.documents) {
          setDocuments(data.documents);
        }
      }
    } catch (error) {
      console.error("Failed to fetch documents", error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        // Trigger parsing automatically
        await fetch(`/api/documents/${data.doc_id}/parse?run_async=true`, {
          method: "POST",
        });
        await fetchDocuments();
      } else {
        alert("Upload failed");
      }
    } catch (error) {
      console.error("Upload error", error);
      alert("Upload error");
    } finally {
      setUploading(false);
      e.target.value = ""; // Reset input
    }
  };

  const triggerParse = async (docId: string) => {
    try {
      const res = await fetch(`/api/documents/${docId}/parse?run_async=true`, {
        method: "POST",
      });
      if (res.ok) {
        alert("Parse task started");
      } else {
        alert("Failed to start parse task");
      }
    } catch (error) {
      console.error("Parse error", error);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Documents Management</h1>
        <p className="subtitle">Upload and parse academic papers</p>
      </div>

      <div className="card">
        <div style={{ marginBottom: "1rem" }}>
          <label className="btn btn-primary" style={{ cursor: "pointer" }}>
            {uploading ? "Uploading..." : "Upload PDF / MD"}
            <input 
              type="file" 
              accept=".pdf,.md,.txt" 
              onChange={handleFileUpload} 
              style={{ display: "none" }} 
              disabled={uploading}
            />
          </label>
        </div>

        {loading ? (
          <div>Loading documents...</div>
        ) : documents.length === 0 ? (
          <div className="empty-state">No documents uploaded yet.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Title</th>
                <th>Year</th>
                <th>Chunks</th>
                <th>Uploaded At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.doc_id}>
                  <td>{doc.filename}</td>
                  <td>{doc.metadata?.title || "-"}</td>
                  <td>{doc.metadata?.year || "-"}</td>
                  <td>{doc.chunk_count || 0}</td>
                  <td>{new Date(doc.upload_time).toLocaleString()}</td>
                  <td>
                    <button 
                      className="btn btn-outline" 
                      onClick={() => triggerParse(doc.doc_id)}
                    >
                      Re-parse
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
