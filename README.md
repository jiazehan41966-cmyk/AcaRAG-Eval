# 面向学术论文知识生产的 Agentic RAG 自动化评测平台

当前版本已经具备完整的研发闭环：

- 文档接入：PDF/Markdown/TXT 上传、解析、chunk 化
- 检索链路：Qdrant + BGE embedding + BM25 + Hybrid + Rerank
- Agent 编排：LangGraph（router/rewrite/retrieve/grade/generate/check）
- 工具访问：`paper-eval-mcp-server` 真实工具调用
- 自动评测：Heuristic + RAGAS + DeepEval 双通道
- 回归机制：baseline 保存、diff 对比、regression markdown
- 可观测：Langfuse（可选）+ 本地 trace 存储
- 前端页：`/ui/evaluation` 最小评测看板

## 快速启动

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

入口：

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Eval UI: [http://localhost:8000/ui/evaluation](http://localhost:8000/ui/evaluation)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

## 关键环境变量

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=paper_chunks
QDRANT_FALLBACK_LOCAL=true

EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base

LLM_PROVIDER=none
OPENAI_API_KEY=

ENABLE_RAGAS=false
ENABLE_DEEPEVAL=false

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

说明：

- Qdrant 不可达时可回退本地 Qdrant（`QDRANT_FALLBACK_LOCAL=true`）。
- 未配置 OpenAI 时，LLM 节点会自动使用启发式兜底，服务仍可运行。
- RAGAS/DeepEval 默认关闭，按需在请求里开启。

## API

### 文档与检索

- `POST /api/documents/upload`
- `POST /api/documents/{doc_id}/parse`
- `POST /api/index/build`
- `POST /api/search/vector`
- `POST /api/search/bm25`
- `POST /api/search/hybrid`
- `POST /api/search/rerank`

### Agent

- `POST /api/chat/ask`

### 评测

- `POST /api/eval/run`
- `GET /api/eval/runs`
- `GET /api/eval/runs/{run_id}`

### Trace

- `GET /api/traces`
- `GET /api/traces/{trace_id}`

## 评测请求示例

```bash
curl -X POST "http://localhost:8000/api/eval/run" \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "baseline_v1",
    "top_k": 5,
    "run_ragas": true,
    "run_deepeval": true,
    "set_as_baseline": true,
    "baseline_name": "paper_qa_main"
  }'
```

再次回归对比：

```bash
curl -X POST "http://localhost:8000/api/eval/run" \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "experiment_v2",
    "top_k": 5,
    "baseline_name": "paper_qa_main",
    "compare_with_baseline": true
  }'
```

## CI

已提供 GitHub Actions：

- `ruff check`
- `pytest`
- 可选 `eval smoke`（通过环境变量 `CI_RUN_EVAL_SMOKE=true` 开启）

工作流文件：`.github/workflows/ci.yml`

## 本地测试

```bash
pytest -q
```
