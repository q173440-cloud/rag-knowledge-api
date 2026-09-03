from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import requests
app = FastAPI()
class AskRequest(BaseModel):
    question: str
    model: str
@app.post("/ask")
def ask(request: AskRequest):
    question = request.question
    model = request.model
    context = retrieve(question, chunks)
    prompt = f"""
        参考资料：
        {context}

        用户问题：
        {question}

        请根据参考资料回答用户问题。
        """
    answer = ask_llm(prompt, model)
    return {
    "answer": answer,
    "sources": [context]
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
        cunch=score_chunk(question,chunk)
        cunchu.append((cunch,chunk))
    cunchu.sort(key=get_score,reverse=True)
    return cunchu[0][1]    
chunks = load_chunks("data/knowledge.txt")


#question = "RAG的核心思想是什么"



#context = retrieve(question, chunks)



#answer = ask_llm(prompt, "gemini-3-flash")

#print(answer)
