from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import json
import requests


from app.retrieval import (
    semantic_retrieve,
    gate,
    chunks,
    chunk_embeddings,
    embedding_model
)


app = FastAPI()


class AskRequest(BaseModel):
    question: str
    model: str
    top_k: int = Field(default=3, ge=1, le=5)


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question
    model = request.model
    top_k = request.top_k
    print("top_k =", top_k)
    top_results = semantic_retrieve(
        question,
        chunks,
        chunk_embeddings,
        embedding_model,
        top_k
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



def ask_llm(prompt, model):
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="API 密钥不存在"
        )

    url = os.getenv("LLM_API_URL")

    if not url:
        raise HTTPException(
            status_code=503,
            detail="API地址不存在"
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            url=url,
            headers=headers,
            json=data,
            timeout=10
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=503,
            detail="大模型请求超时"
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="无法连接大模型服务"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail="大模型服务返回错误"
        )

    shuju = json.loads(response.text)
    answer = shuju["choices"][0]["message"]["content"]
    return answer
