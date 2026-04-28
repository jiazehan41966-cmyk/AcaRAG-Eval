# AcaRAG Eval Next Dashboard

Minimal React/Next.js dashboard for the Agentic RAG evaluation platform.

```bash
cd frontend/next-dashboard
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Pages:

- `/` dashboard for documents, latest metrics, and eval runs.
- `/evaluation-set` local JSONL Golden Set inspection and lightweight eval trigger.
