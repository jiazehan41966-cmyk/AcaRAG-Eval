"use client";

import { useState } from "react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: any[];
  trace?: any;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const res = await fetch("/api/chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage, top_k: topK }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setMessages(prev => [
          ...prev, 
          { 
            role: "assistant", 
            content: data.answer,
            citations: data.citations,
            trace: data.trace
          }
        ]);
      } else {
        const err = await res.json();
        setMessages(prev => [...prev, { role: "assistant", content: `Error: ${err.detail || "Request failed"}` }]);
      }
    } catch (error) {
      console.error("Chat error", error);
      setMessages(prev => [...prev, { role: "assistant", content: "Error connecting to server." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 4rem)" }}>
      <div className="header" style={{ flexShrink: 0 }}>
        <h1>Academic Chat</h1>
        <p className="subtitle">Agentic RAG with citations</p>
      </div>

      <div style={{ marginBottom: "1rem", display: "flex", gap: "1rem", alignItems: "center" }}>
        <label>Top K Chunks:</label>
        <input 
          type="number" 
          value={topK} 
          onChange={e => setTopK(Number(e.target.value))} 
          style={{ width: "60px", padding: "0.25rem" }}
          min={1}
          max={20}
        />
      </div>

      <div className="card" style={{ flexGrow: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem", marginBottom: "1rem" }}>
        {messages.length === 0 ? (
          <div className="empty-state">Ask a question about the uploaded papers to start...</div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} style={{ 
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "80%",
              backgroundColor: msg.role === "user" ? "var(--primary-color)" : "var(--surface)",
              color: msg.role === "user" ? "white" : "inherit",
              padding: "1rem",
              borderRadius: "0.5rem",
              border: msg.role === "user" ? "none" : "1px solid var(--border-color)"
            }}>
              <div style={{ whiteSpace: "pre-wrap", marginBottom: msg.citations?.length ? "1rem" : "0" }}>
                {msg.content}
              </div>
              
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ fontSize: "0.85rem", borderTop: "1px solid var(--border-color)", paddingTop: "0.5rem", marginTop: "0.5rem" }}>
                  <strong>Citations:</strong>
                  <ul style={{ paddingLeft: "1.5rem", marginTop: "0.25rem" }}>
                    {msg.citations.map((cit, cidx) => (
                      <li key={cidx}>
                        Doc: {cit.doc_id || "Unknown"} (Page {cit.page || "?"}) 
                        - Score: {cit.score}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))
        )}
        {loading && <div style={{ alignSelf: "flex-start" }}>Agent is thinking...</div>}
      </div>

      <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0 }}>
        <input 
          type="text" 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          onKeyDown={e => e.key === "Enter" && handleSend()}
          placeholder="Ask a question..."
          style={{ flexGrow: 1, padding: "0.75rem", borderRadius: "0.25rem", border: "1px solid var(--border-color)" }}
          disabled={loading}
        />
        <button className="btn btn-primary" onClick={handleSend} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
