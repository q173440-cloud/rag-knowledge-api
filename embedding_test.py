from app.retrieval import cosine_similarity, embedding_model

texts = [
    "RAG 会在回答前查询外部资料。",
    "模型回答问题之前先寻找外部参考信息。",
    "HTTP 用于客户端与服务器通信。"
]

embeddings = embedding_model.encode(texts)

score_ab = cosine_similarity(embeddings[0], embeddings[1])
score_ac = cosine_similarity(embeddings[0], embeddings[2])

print("A 和 B：", score_ab)
print("A 和 C：", score_ac)