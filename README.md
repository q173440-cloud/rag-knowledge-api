# RAG Knowledge API

一个基于 FastAPI、Semantic Retrieval（语义检索）、远程 Embedding API 和 LLM API 构建的最小 RAG 知识库问答项目。

当前项目支持：

- 本地文本知识库
- 远程 OpenAI-compatible Embedding API（默认使用硅基流动 `BAAI/bge-m3`）
- Top-K Semantic Retrieval（支持请求参数配置 `top_k`）
- 相似度阈值过滤
- Multi-Chunk Context
- No-Match Gate
- 自动从项目根目录 `.env` 加载配置
- FastAPI `/ask` 接口（默认模型 `deepseek-flash`）
- Retrieval Evaluation

---

## Project Structure

```text
rag-knowledge-api/
├─ app/
│  ├─ main.py
│  └─ retrieval.py
├─ data/
│  └─ knowledge.txt
├─ eval_retrieval.py
├─ embedding_test.py
├─ verify_top_k.py
├─ requirements.txt
├─ .env.example
├─ AGENTS.md
├─ .gitignore
└─ README.md
```

主要文件：

- `app/main.py`：FastAPI 接口、RAG 请求流程和 LLM 调用
- `app/retrieval.py`：`.env` 加载、知识库加载、远程 Embedding 调用、Semantic Retrieval 和 Gate
- `data/knowledge.txt`：本地知识库
- `eval_retrieval.py`：Retrieval 回归测试
- `embedding_test.py`：远程 Embedding 相似度测试脚本
- `verify_top_k.py`：`top_k` 参数验证脚本
- `.env.example`：环境变量配置模板（复制为 `.env` 后填入真实密钥）
- `AGENTS.md`：项目安全规范（禁止提交任何密钥或凭据）

---

## RAG Flow

```text
用户问题
↓
Remote Embedding API (Semantic Retrieval)
↓
Top-K 候选资料
↓
Similarity Threshold Filtering
↓
Multi-Chunk Context
↓
LLM (默认 deepseek-flash)
↓
最终回答
```

当前使用：

- Embedding Model：`BAAI/bge-m3`（通过硅基流动 Embeddings API 调用）
- LLM Model：`deepseek-flash`（可通过请求参数 `model` 或 `.env` 中的 `LLM_MODEL` 覆盖）
- Top-K：默认 `3`（支持 `1`~`5`）
- Similarity Threshold：`0.30`

只有达到当前阈值的资料才会真正进入 Context。

如果最高相似度低于 `0.30`：

```text
answer = 知识库没有足够相关资料
sources = []
```

并且不会调用 LLM。

`0.30` 是之前针对小型知识库和旧模型使用的临时阈值，不是通用标准。

---

## Environment

当前项目使用 Python，可在 Linux 或 Windows 下运行。

创建虚拟环境（Linux/macOS）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

创建虚拟环境（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Configuration (`.env`)

复制 `.env.example` 为 `.env`，并填入真实密钥：

```text
# LLM 配置
LLM_API_KEY=your-llm-api-key
LLM_API_URL=https://api.deepseek.com/v1/chat/completions
LLM_MODEL=deepseek-flash

# Embedding 向量模型配置（硅基流动）
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL=BAAI/bge-m3
```

说明：

- 程序导入 `app.retrieval` 时会自动从项目根目录的 `.env` 加载上述环境变量（若系统已提前导出同名环境变量，则优先使用系统环境变量）。
- `.env` 与 `.env.*`（除 `.env.example` 外）已在 `.gitignore` 中忽略。**不要把任何真实 API Key 写入代码、README、模板或提交到 Git。**
- 当前 LLM 与 Embedding 均使用 OpenAI-compatible 接口格式。

---

## Run Retrieval Evaluation

Retrieval Evaluation 需要配置 `EMBEDDING_API_KEY`（不需要配置 `LLM_API_KEY`）。

在项目根目录运行：

```bash
python eval_retrieval.py
```

当前测试主要覆盖：

- 明确命中
- 语义改写
- No-Match
- Multi-Chunk Retrieval

切换为 `BAAI/bge-m3` 后的当前测试结果（`threshold = 0.30`）：

```text
Case A | PASS
Case B | PASS
Case C | FAIL
Case D | PASS

Passed: 3 / 4
```

Case C（无关问题）中，由于 `BAAI/bge-m3` 整体余弦相似度分值高于原本地模型，`0.30` 的临时阈值未能将其拦截，因此测试失败。当前保留这个真实结果，便于后续按新模型分布重新标定阈值。

---

## Start API

在配置好 `.env` 后，于项目根目录运行：

```bash
python -m uvicorn app.main:app --reload
```

默认地址：

```text
http://127.0.0.1:8000
```

健康检查：

```text
GET /health
```

FastAPI 文档：

```text
http://127.0.0.1:8000/docs
```

---

## POST /ask

主要接口：

```text
POST /ask
```

请求示例（`model` 与 `top_k` 为可选字段）：

```json
{
  "question": "RAG 的核心思想是什么？"
}
```

也可以显式指定 `model` 和 `top_k`：

```json
{
  "question": "RAG 的核心思想是什么？",
  "model": "deepseek-flash",
  "top_k": 3
}
```

响应示例：

```json
{
  "answer": "根据参考资料生成的回答",
  "sources": [
    "本次实际提供给 LLM 的知识库资料"
  ],
  "best_score": 0.68
}
```

其中：

- `answer`：LLM 根据 Context 生成的回答
- `sources`：本次真正进入 Context 的资料
- `best_score`：排名第一候选资料的语义相似度

如果知识库没有足够相关资料，则不会调用 LLM。

如果知识库暂无可用资料（如知识库为空或仅包含空白内容）：

```json
{
  "answer": "知识库暂无可用资料",
  "sources": [],
  "best_score": null
}
```

并且不会进行 Embedding 检索或调用 LLM。

---

## Custom Knowledge Base

当前知识库存放在：

```text
data/knowledge.txt
```

如果希望使用自己的资料，可以修改或替换这个文件。

修改知识库以后，需要重新启动程序，使新的知识内容重新生成 Embedding。

### Evaluation 注意事项

`eval_retrieval.py` 中的测试问题和 `expected_sources` 是针对仓库自带的 `knowledge.txt` 编写的。

因此，如果替换了 `knowledge.txt`，原来的 Evaluation 测试案例可能不再适用，需要根据新的知识库重新设计测试问题和预期资料。

---

## Current Limitations

当前项目仍然存在以下限制：

- 知识库规模较小
- 当前只支持本地文本知识库
- Embedding 在内存中进行检索
- 尚未使用 Vector Database
- `0.30` 是临时 Similarity Threshold（尚待针对 `BAAI/bge-m3` 重新校准）
- Top-K 默认值为 3（支持 `1`~`5`）
- Evaluation Case 数量较少
- Semantic Similarity 高不代表资料一定能够回答问题
- Context 中存在相关资料也不能保证 LLM 最终回答一定正确
- 暂未支持 PDF 等复杂文档输入

当前项目用于实现和验证最小 RAG 工程流程，不是生产级 RAG 系统。
