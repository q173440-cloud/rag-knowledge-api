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