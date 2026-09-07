# RAG Knowledge API

一个使用 FastAPI、简单文本检索和 LLM API 构建的最小 RAG 知识库问答项目。

## 功能

- 从本地 `knowledge.txt` 读取知识
- 将知识文档切分成多个 chunk
- 根据用户问题检索相关 chunk
- 将检索结果作为 Context 提供给 LLM
- 通过 FastAPI 提供 `/ask` HTTP 接口
- 返回模型回答和检索到的 source

## 数据流

```text
knowledge.txt
↓
load_chunks()
↓
chunks

question
↓
retrieve()
↓
context
↓
context + question
↓
prompt
↓
LLM API
↓
answer
↓
FastAPI Response
启动

设置环境变量：

LLM_API_KEY

然后在项目根目录运行：

C:\Users\wtqweeqopt\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn app.main:app --reload

打开：

http://127.0.0.1:8000/docs
API 示例

请求：

{
  "question": "RAG的核心思想是什么",
  "model": "gemini-3-flash"
}

响应示例：

{
  "answer": "根据参考资料生成的回答",
  "sources": [
    "Retrieval 检索到的知识库文本"
  ]
}
当前 Retrieval

当前版本使用非常简单的字符重合数量作为相关性分数。

question 与 chunk 相同字符越多
→ score 越高
→ 优先返回高分 chunk

这是用于建立 RAG 基础数据流的 baseline，并不具备真正的语义理解能力。

当前局限
字符重合不等于语义相关
知识库没有答案时仍然会返回一个最高分 chunk
暂未使用 Embedding
暂未使用向量数据库
暂未支持 PDF 等复杂文档

后续版本将逐步改进 Retrieval。

## Day 6 - No-Match Gate

在原有 lexical retrieval baseline 上增加了一个简单的 No-Match Gate。

当前流程：

question
→ retrieval
→ best_score + best_chunk
→ exact-term gate
→ generation

当问题和检索结果之间没有明确主题词匹配时：

- 不调用 LLM
- 返回“知识库没有足够相关资料”
- sources 返回空列表

### 当前限制

当前 Gate 只是教学用的关键词规则，不能判断知识库是否真的存在答案。

例如：

`FastAPI 默认使用什么数据库？`

虽然问题和检索资料都包含 `FastAPI`，Gate 会放行，但检索到的资料实际上并没有数据库相关答案。

因此：

关键词匹配成功 ≠ 资料一定足够回答问题。

## Semantic Retrieval

项目已经从基于字符重合的 Lexical Retrieval，
升级为基于 Embedding 的 Semantic Retrieval。

当前使用：

- sentence-transformers
- paraphrase-multilingual-MiniLM-L12-v2

基本流程：

用户问题
→ Question Embedding
→ 与预计算的 Chunk Embeddings 比较
→ Semantic Similarity
→ 选择最高分 Chunk
→ 作为 Context 交给 LLM

知识库 Chunk Embeddings 会在程序启动时预计算，
避免每次用户提问都重新计算。

### 当前临时 Gate

根据当前模型、知识库和少量测试数据，
暂时使用 0.30 作为最低语义相似度。

该阈值只是当前实验 baseline，不是通用标准。

### 当前限制

Semantic similarity 高
不代表资料一定包含回答问题所需的具体信息。

例如：

FastAPI 默认使用什么数据库？

虽然可以检索到 FastAPI 相关资料，
但资料本身可能并没有数据库相关答案。

## Top-K Semantic Retrieval 与 Multi-Chunk Context

项目目前支持 Top-K Semantic Retrieval。

Semantic Retrieval 会计算用户问题与知识库各个 chunk 的语义相似度，按照相似度从高到低排序，并取得前 K 个候选结果。

```python
top_results = results[:top_k]
```

其中：

- `top_k`：最多取得多少个候选资料
- `top_results`：排名前 K 的候选结果

### Candidate Filtering

Top-K 只是候选资料，不代表所有候选都会提供给 LLM。

当前项目使用临时相似度阈值 `0.30` 进行筛选：

```python
context_chunks = []

for result in top_results:
    if result[0] >= 0.3:
        context_chunks.append(result[1])
    else:
        break
```

只有满足当前相关度要求的资料才会真正进入 Context。

### Multi-Chunk Context

筛选后的多个 chunk 会组合成一段 Context：

```python
context_text = "\n\n".join(context_chunks)
```

这样一个问题可以同时参考多个相关知识库片段，而不再只能使用 Top-1 的单条资料。

### Sources

API 返回的 `sources` 表示本次真正进入 Context、实际提供给 LLM 的资料。

例如：

```json
{
  "sources": [
    "资料A",
    "资料B"
  ]
}
```

这样可以保留每条资料之间的边界，方便后续程序、前端和调试使用。

### No-Match Gate

如果最高相似度：

```text
best_score < 0.30
```

则直接返回知识库没有足够相关资料，并且不会调用 LLM。

### 当前已验证

- 明确命中问题可以正常检索
- 语义改写问题仍可以找到相关资料
- 明显无关问题可以被 Gate 拦截
- 一个问题需要多个 chunk 时，可以同时使用多个相关资料
- `sources` 与实际进入 Context 的资料保持一致

### 当前限制

`0.30` 只是当前 Embedding 模型和小型知识库下使用的临时阈值，并不是通用标准。

同时：

- 语义相似度高，不代表资料一定包含问题答案
- Top-K 找到多个相关资料，不代表知识库一定能够回答问题
- Context 中存在相关资料，也不能保证 LLM 最终回答一定正确

当前阶段主要验证 Retrieval、Candidate Filtering、Multi-Chunk Context 和 sources 的数据流，尚未进行正式的回答质量 Evaluation。

## Minimal Retrieval Evaluation

新增了一个最小 Retrieval Evaluation，用于在修改检索逻辑后快速检查已有能力是否发生退化。

运行：

```powershell
python eval_retrieval.py

当前覆盖：

明确命中
语义改写
No-Match
Multi-Chunk

当前结果：

Case A | PASS
Case B | FAIL
Case C | PASS
Case D | PASS

Passed: 3 / 4

Case B 中，预期的 RAG 核心资料在语义相似度排序中只排第 5，因此没有进入 Top-3。当前仅记录这个限制，没有为了让测试通过而调整 threshold 或 Top-K。

当前测试集只有少量固定案例，不代表完整的 RAG benchmark，也不能证明 Retrieval 对所有现实问题都可靠。