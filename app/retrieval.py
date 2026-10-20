from sentence_transformers import SentenceTransformer, util


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        chunks = text.split("\n\n")
        return chunks


def get_score(item):
    return item[0]


def semantic_retrieve(
    question,
    chunks,
    chunk_embeddings,
    embedding_model,
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

    top_results = results[:top_k]
    # 排名前 K 的候选结果

    return top_results


def gate(best_score):
    if best_score < 0.3:
        return False
    else:
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

