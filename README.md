# RAG Knowledge API

一个基于 FastAPI、Semantic Retrieval（语义检索）和 LLM API 构建的最小 RAG 知识库问答项目。

当前项目支持：

- 本地文本知识库
- Sentence Transformers Embedding
- Top-K Semantic Retrieval
- 相似度阈值过滤
- Multi-Chunk Context
- No-Match Gate
- FastAPI `/ask` 接口
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
├─ requirements.txt
├─ .gitignore
└─ README.md
```

主要文件：

- `app/main.py`：FastAPI 接口、RAG 请求流程和 LLM 调用
- `app/retrieval.py`：知识库加载、Embedding、Semantic Retrieval 和 Gate
- `data/knowledge.txt`：本地知识库
- `eval_retrieval.py`：Retrieval 回归测试

---

## RAG Flow

```text
用户问题
↓
Semantic Retrieval
↓
Top-K 候选资料
↓
Similarity Threshold Filtering
↓
Multi-Chunk Context
↓
LLM
↓
最终回答
```

当前使用：

- Embedding Model：`paraphrase-multilingual-MiniLM-L12-v2`
- Top-K：最多取得 3 个候选资料
- Similarity Threshold：`0.30`

只有达到当前阈值的资料才会真正进入 Context。

如果最高相似度低于 `0.30`：

```text
answer = 知识库没有足够相关资料
sources = []
```

并且不会调用 LLM。

`0.30` 只是针对当前模型、小型知识库和现有测试数据使用的临时阈值，不是通用标准。

---

## Environment

当前项目使用 Python。

本项目目前在 Windows + PowerShell 环境下开发和验证。

创建虚拟环境：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

---

## LLM Configuration

启动 API 前需要配置两个环境变量：

```text
LLM_API_KEY
LLM_API_URL
```

其中：

- `LLM_API_KEY`：LLM API 的访问密钥
- `LLM_API_URL`：LLM API 请求地址

PowerShell 临时设置方式：

```powershell
$env:LLM_API_KEY="your-api-key"
$env:LLM_API_URL="your-api-url"
```

不要把真实 API Key 写入代码、README 或提交到 Git。

当前 LLM 请求使用 OpenAI-compatible Chat Completions 风格的接口格式。

调用 `/ask` 时还需要提供 `model`，模型名称必须能够被当前配置的 LLM API 服务识别。

---

## Run Retrieval Evaluation

Retrieval Evaluation 不需要配置 `LLM_API_KEY` 和 `LLM_API_URL`。

在项目根目录运行：

```powershell
python eval_retrieval.py
```

当前测试主要覆盖：

- 明确命中
- 语义改写
- No-Match
- Multi-Chunk Retrieval

当前测试结果：

```text
Case A | PASS
Case B | FAIL
Case C | PASS
Case D | PASS

Passed: 3 / 4
```

Case B 中，预期的 RAG 核心资料没有进入当前 Top-3，因此测试失败。

当前保留这个真实失败结果，没有为了让测试全部通过而专门调整 threshold 或 Top-K。

当前 Evaluation 只有少量固定案例，用于检查 Retrieval 修改后是否发生明显退化，不能代表完整的 RAG Benchmark。

---

## Start API

首先配置：

```text
LLM_API_KEY
LLM_API_URL
```

然后在项目根目录运行：

```powershell
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

请求示例：

```json
{
  "question": "RAG 的核心思想是什么？",
  "model": "your-model-name"
}
```

响应示例：

```json
{
  "answer": "根据参考资料生成的回答",
  "sources": [
    "本次实际提供给 LLM 的知识库资料"
  ],
  "best_score": 0.42
}
```

其中：

- `answer`：LLM 根据 Context 生成的回答
- `sources`：本次真正进入 Context 的资料
- `best_score`：排名第一候选资料的语义相似度

如果知识库没有足够相关资料，则不会调用 LLM。

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
- `0.30` 是临时 Similarity Threshold
- Top-K 当前固定为 3
- Evaluation Case 数量较少
- Semantic Similarity 高不代表资料一定能够回答问题
- Context 中存在相关资料也不能保证 LLM 最终回答一定正确
- 暂未支持 PDF 等复杂文档输入

当前项目用于实现和验证最小 RAG 工程流程，不是生产级 RAG 系统。