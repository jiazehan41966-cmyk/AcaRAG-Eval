# 面向学术论文知识生产的 Agentic RAG 自动化评测平台

这是项目的第一阶段可运行版本（MVP），目标是先跑通闭环：

- 文档上传（PDF / Markdown / TXT）
- 文档解析与 chunk 切分
- 索引构建（vector / BM25 / hybrid）
- 检索接口与 rerank 接口
- 最小 Agentic RAG 问答流程（router -> retrieve -> generate -> faithfulness check）

> 当前版本为了快速启动，使用本地轻量实现（哈希 embedding + 本地 BM25）。后续可平滑替换为 Qdrant + BGE + LangGraph + RAGAS/DeepEval。

## 目录结构

```text
backend/
  app/
    api/
    core/
    models/
    schemas/
    services/
    workflows/
mcp_servers/
frontend/
data/
docker-compose.yml
requirements.txt
```

## 快速启动

### 1) 本地运行

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开文档：

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

### 2) Docker Compose 运行

```bash
docker compose up --build
```

### 3) 运行测试

```bash
pytest
```

## 已实现接口

### 文档接入层

- `POST /api/documents/upload`
- `GET /api/documents/{doc_id}`
- `POST /api/documents/{doc_id}/parse`
- `GET /api/documents/{doc_id}/chunks`
- `GET /api/documents/{doc_id}/metadata`

### 知识索引层

- `POST /api/index/build`
- `POST /api/index/rebuild`
- `POST /api/search/vector`
- `POST /api/search/bm25`
- `POST /api/search/hybrid`
- `POST /api/search/rerank`

### Agentic RAG 工作流层（MVP）

- `POST /api/chat/ask`

### 自动化评测层（占位）

- `POST /api/eval/run`
- `GET /api/eval/runs/{run_id}`

### 可观测性层（占位）

- `GET /api/traces/{trace_id}`

## 示例调用

### 1) 上传文档

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./example.md"
```

### 2) 解析文档

```bash
curl -X POST "http://localhost:8000/api/documents/{doc_id}/parse" \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 600, "chunk_overlap": 120}'
```

### 3) 构建索引

```bash
curl -X POST "http://localhost:8000/api/index/build" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 4) 混合检索

```bash
curl -X POST "http://localhost:8000/api/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query":"这篇论文的方法和实验设置是什么？","top_k":5}'
```

### 5) Agentic RAG 问答

```bash
curl -X POST "http://localhost:8000/api/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"请总结该论文的核心贡献","top_k":5}'
```

## 下一阶段计划

1. 接入 Qdrant 与真实 Embedding/Reranker（BGE 系列）
2. 用 LangGraph 替换当前轻量 Agent 工作流
3. 增加 Query Rewrite、文档相关性评分与二次检索
4. 接入 RAGAS + DeepEval 批量评测与报告导出
5. 接入 Langfuse trace 与 token/cost 统计
6. 完成 `paper-eval-mcp-server` 的 MCP 工具能力
