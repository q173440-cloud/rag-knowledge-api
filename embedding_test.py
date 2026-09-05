from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [
    "RAG 会在回答前查询外部资料。",
    "模型回答问题之前先寻找外部参考信息。",
    "HTTP 用于客户端与服务器通信。"
]

embeddings = model.encode(
    texts,
    convert_to_tensor=True
)

score_ab = util.cos_sim(
    embeddings[0],
    embeddings[1]
).item()

score_ac = util.cos_sim(
    embeddings[0],
    embeddings[2]
).item()

print("A 和 B：", score_ab)
print("A 和 C：", score_ac)