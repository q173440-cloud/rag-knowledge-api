from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import requests

from app.retrieval import (
    semantic_retrieve,
    gate,
    chunks,
    chunk_embeddings,
    embedding_model,
    get_score
)


topic_terms = [
    "Python",
    "FastAPI",
    "HTTP",
    "JSON",
    "RAG",
    "Retrieval",
    "Context",
    "Embedding",
    "Generation"
]


app = FastAPI()


class AskRequest(BaseModel):
    question: str
    model: str


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question
    model = request.model

    top_results = semantic_retrieve(
        question,
        chunks,
        chunk_embeddings,
        embedding_model
    )

    best_score = top_results[0][0]
    # 第一名候选的相似度

    matched = gate(best_score)
    # 使用 retrieval.py 中的 gate

    if matched == False:
        return {
            "answer": "知识库没有足够相关资料",
            "sources": [],
            "best_score": best_score
        }

    context_chunks = []

    for result in top_results:
        if result[0] >= 0.3:
            context_chunks.append(result[1])
        else:
            break

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
        参考资料：
        {context_text}

        用户问题：
        {question}

        请根据参考资料回答用户问题。
        """

    answer = ask_llm(prompt, model)

    return {
        "answer": answer,
        "sources": context_chunks,
        "best_score": best_score
    }


@app.get("/health")
def ceshi():
    return {"status": "ok"}


api_key = os.environ["LLM_API_KEY"]

url = "https://api.00033333.xyz/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


def ask_llm(prompt, model):
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        url=url,
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        print("请求失败")
        print(response.status_code)
        print(response.text)
        return

    shuju = json.loads(response.text)
    answer = shuju["choices"][0]["message"]["content"]

    return answer


# -------------------------
# 以下是以前的 Lexical Retrieval 旧代码
# Day 10 暂时不进行历史代码清理
# -------------------------

def score_chunk(question, chunk):
    score = 0

    for chur in question:
        if chur in chunk:
            score += 1

    return score


def retrieve(question, chunks):
    cunchu = []

    for chunk in chunks:
        cunch = score_chunk(question, chunk)
        cunchu.append((cunch, chunk))

    cunchu.sort(key=get_score, reverse=True)

    best_score = cunchu[0][0]
    best_chunk = cunchu[0][1]

    return (best_score, best_chunk)