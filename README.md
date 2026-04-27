# 面向学术论文知识生产的 Agentic RAG 自动化评测平台

当前版本已形成从检索到评测再到回归门禁的工程闭环：

- 文档接入：PDF/Markdown/TXT 上传、解析、chunk 化
- 检索链路：Qdrant + BGE embedding + BM25 + Hybrid + Rerank
- Agent 编排：LangGraph（router/rewrite/retrieve/grade/generate/check）
- MCP 工具：标准 FastMCP `paper-eval-mcp-server`
- 自动评测：Heuristic + RAGAS + DeepEval 双通道
- 回归机制：baseline 保存、diff 对比、regression gate
- 可观测：Langfuse（可选）+ 本地 trace 存储
- 前端页：`/ui/evaluation`（手动触发评测、run 列表、case trace/citation 详情）

## 1) 本地快速启动

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

## 2) `.env` 模板说明

基于 `.env.example` 创建 `.env`，至少关注以下项：

```env
# Vector retrieval
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=paper_chunks
QDRANT_FALLBACK_LOCAL=true

# Embedding / rerank
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base

# LLM + eval channels
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=
ENABLE_RAGAS=true
ENABLE_DEEPEVAL=true

# MCP transport
MCP_MODE=inprocess
MCP_SERVER_URL=http://127.0.0.1:8765/mcp
MCP_TIMEOUT_SECONDS=15

# Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_TRACING_ENABLED=true
```

说明：

- `MCP_MODE=inprocess`：后端直接以内嵌 FastMCP server 调用工具。
- `MCP_MODE=http`：后端通过 `MCP_SERVER_URL` 连接独立部署的 FastMCP 服务。
- 未配置 `OPENAI_API_KEY` 时，LLM 节点和 DeepEval 会自动降级/跳过。

## 3) Docker 一键启动

```bash
docker compose up --build
```

该命令会同时启动：

- backend（FastAPI）
- qdrant
- redis
- postgres

如需后台运行：

```bash
docker compose up -d --build
```

## 4) 标准 FastMCP 服务部署

### 4.1 以 stdio 方式运行（本地开发）

```bash
python -m mcp_servers.paper_eval_server.server --transport stdio
```

### 4.2 以 HTTP 方式运行（独立部署）

```bash
python -m mcp_servers.paper_eval_server.server --transport streamable-http --host 0.0.0.0 --port 8765 --path /mcp
```

后端切换为远程 MCP：

```env
MCP_MODE=http
MCP_SERVER_URL=http://127.0.0.1:8765/mcp
```

## 5) 评测与回归

### 5.1 扩展 Golden Set

```bash
python scripts/generate_golden_set.py
```

默认生成 `80` 条用例（fact/comparison/multi-hop/metadata）。

### 5.2 运行评测

```bash
curl -X POST "http://localhost:8000/api/eval/run" \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "baseline_v1",
    "top_k": 5,
    "run_ragas": true,
    "run_deepeval": true,
    "set_as_baseline": true,
    "baseline_name": "paper_main"
  }'
```

### 5.3 失败归因阈值校准

```bash
python scripts/calibrate_failure_thresholds.py --write
```

阈值写入：`backend/config/failure_thresholds.json`。

### 5.4 回归门禁

```bash
python scripts/check_regression_gate.py --latest
```

门禁规则：`backend/config/regression_gate.json`。

## 6) 真实环境验收（OPENAI + RAGAS + DeepEval + Langfuse）

```bash
python scripts/run_real_acceptance.py --run-name real_acceptance --max-cases 20
```

产物：

- `data/eval_reports/acceptance_*.md`
- `data/eval_reports/acceptance_*.png`

说明：脚本会在缺少 OPENAI/Langfuse 关键配置时返回非零退出码，并在报告中标记阻塞原因。

## 7) CI

工作流：`.github/workflows/ci.yml`

包含：

- `ruff check`
- `pytest`
- `python scripts/check_regression_gate.py --latest`（指标回归门禁）

## 8) 常见故障排查

### Q1: DeepEval 显示 skipped / OPENAI_API_KEY missing

- 检查 `.env` 的 `OPENAI_API_KEY` 是否配置。
- 检查 `LLM_PROVIDER=openai`。
- 检查模型可用性（默认 `gpt-4o-mini`）。

### Q2: Langfuse enabled=false

- 检查 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`。
- 检查 `LANGFUSE_HOST` 是否可访问。

### Q3: Qdrant 连接失败

- 确认 `QDRANT_URL`。
- 本地开发可设置 `QDRANT_FALLBACK_LOCAL=true`。

### Q4: 回归门禁失败

- 先查看最新 run 的 `baseline_comparison`。
- 对照 `backend/config/regression_gate.json` 调整门限。
- 必要时重新设定 baseline run。

### Q5: MCP HTTP 模式调用失败

- 确认 FastMCP server 已启动且 URL 与 `MCP_SERVER_URL` 一致。
- 先用 `MCP_MODE=inprocess` 验证业务逻辑，再切 HTTP。

---

建议开发顺序：先保证文档入库与检索质量，再做 Agent 策略优化，最后用评测与回归门禁固定质量。
