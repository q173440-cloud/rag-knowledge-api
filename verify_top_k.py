from app.retrieval import (
    semantic_retrieve,
    chunks,
    chunk_embeddings,
    embedding_model
)

question = "测试"
top_k = 5
top_results = semantic_retrieve(
        question,
        chunks,
        chunk_embeddings,
        embedding_model,
        top_k
    )

print(len(top_results))
