# 面向学术论文知识生产的 Agentic RAG 自动化评测平台

当前版本已实现三条主线：

- 真实向量检索：`Qdrant + BGE embedding + BGE reranker`
- Agent 编排：`LangGraph` 状态图（router/rewrite/retrieve/grade/generate/check）
- 自动化评测闭环：`Golden Set + RAGAS(可选) + Markdown 报告导出`

## 核心能力

### 1) 数据接入

- 上传 `PDF/Markdown/TXT`
- 解析与 chunk 切分
- 元数据提取与持久化

### 2) 知识索引

- 向量索引：Qdrant collection `paper_chunks`
- 稀疏索引：本地 BM25
- Hybrid Search：向量 + BM25 融合
- Rerank：BGE Cross-Encoder（不可用时自动降级）

### 3) Agentic RAG

LangGraph 节点：

- `classify_question`
- `rewrite_query`
- `retrieve_docs`
- `grade_documents`
- `generate_answer`
- `check_faithfulness`
- `call_mcp_tool`（当前为占位实现）

### 4) 自动化评测

- 读取 Golden Set（默认 `data/golden_set/golden_set.jsonl`）
- 批量执行问答
- 输出 summary metrics（faithfulness/answer_relevancy/context_recall/citation_accuracy）
- 可选执行 RAGAS
- 导出 Markdown 报告到 `data/eval_reports/<run_id>.md`

## 快速启动

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

文档地址：

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

## 关键环境变量

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=paper_chunks
QDRANT_FALLBACK_LOCAL=true

EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
FORCE_MOCK_EMBEDDING=false

LLM_PROVIDER=none
OPENAI_API_KEY=

DEFAULT_GOLDEN_SET_PATH=data/golden_set/golden_set.jsonl
ENABLE_RAGAS=false
```

说明：

- 若 `QDRANT_URL` 不可达且 `QDRANT_FALLBACK_LOCAL=true`，会自动回退到本地 Qdrant。
- 若未配置 OpenAI 或模型不可用，会自动使用启发式 rewrite/generate/rerank 兜底。

## 接口

### 文档与索引

- `POST /api/documents/upload`
- `POST /api/documents/{doc_id}/parse`
- `POST /api/index/build`
- `POST /api/index/rebuild`
- `POST /api/search/vector`
- `POST /api/search/bm25`
- `POST /api/search/hybrid`
- `POST /api/search/rerank`

### Agent 与评测

- `POST /api/chat/ask`
- `POST /api/eval/run`
- `GET /api/eval/runs/{run_id}`

## 示例：运行评测

```bash
curl -X POST "http://localhost:8000/api/eval/run" \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "baseline_eval",
    "top_k": 5,
    "run_ragas": false
  }'
```

返回 `report_path` 后可直接打开对应 Markdown 报告。

## 测试

```bash
pytest
```
