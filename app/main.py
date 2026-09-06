from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import requests
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
    # 继续使用昨天的 gate

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
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        chunks = text.split("\n\n")
        return chunks

def get_score(item):
    return item[0]

def score_chunk(question, chunk):
    score = 0
    for chur in question:
        if chur in chunk:
            score+=1
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

def semantic_retrieve(
    question,           # 用户问题
    chunks,             # 知识库原始文字
    chunk_embeddings,   # 所有知识库资料的语义向量
    embedding_model,    # 把文字转换成语义向量的模型
    top_k=3
):
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )
    # 把用户问题转换成语义向量

    results = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):

        similarity = util.cos_sim(
            question_embedding,
            chunk_embedding
        ).item()
        # 计算“用户问题”和“当前资料”的语义相似度

        results.append((similarity, chunk))
        # 保存：
        # (语义相似度, 当前原始资料)

    results.sort(key=get_score, reverse=True)
    # 按语义相似度从高到低排序
    top_results= results[:top_k]  
    # 排名前 K 的候选结果 
    return top_results
def gate(best_score):
    if best_score <0.3 :
        return False
    else :
        return True

chunks = load_chunks("data/knowledge.txt")
# 读取知识库原始文字资料

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
# 加载 Embedding 模型
# 负责把文字转换成可以比较语义的数字向量

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True
)
# 把所有知识库资料提前转换成语义向量
# 这些资料没有变，所以程序启动时计算一次即可        