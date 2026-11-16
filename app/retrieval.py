import math
import os
from pathlib import Path
import requests


def load_env(env_path: Path) -> None:
    """从 .env 文件加载环境变量（不覆盖已存在的环境变量）。"""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def load_chunks(path):
    valid_chunks = []
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        chunks = text.split("\n\n")
        for chunk in chunks:
            if chunk.strip():
                valid_chunks.append(chunk)
        return valid_chunks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"向量维度不匹配: {len(vec_a)} != {len(vec_b)}")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RemoteEmbeddingModel:
    """通过 OpenAI-compatible Embeddings API（如硅基流动）生成文本向量。"""

    def __init__(self, api_key: str, api_url: str, model_name: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def encode(self, texts, convert_to_tensor: bool = False):
        del convert_to_tensor  # 兼容原有调用签名
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY 未配置")
        if not self.api_url:
            raise RuntimeError("EMBEDDING_API_URL 未配置")

        single_input = isinstance(texts, str)
        inputs = [texts] if single_input else list(texts)
        if not inputs:
            return []

        response = requests.post(
            url=self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": inputs,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding API 请求失败 ({response.status_code}): {response.text}"
            )

        items = sorted(response.json()["data"], key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in items]
        return embeddings[0] if single_input else embeddings


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

        similarity = cosine_similarity(
            question_embedding,
            chunk_embedding
        )
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


BASE_DIR = Path(__file__).resolve().parent.parent
load_env(BASE_DIR / ".env")

knowledge_path = BASE_DIR / "data" / "knowledge.txt"
chunks = load_chunks(knowledge_path)
# 读取知识库原始文字资料

embedding_model = RemoteEmbeddingModel(
    api_key=os.getenv("EMBEDDING_API_KEY", ""),
    api_url=os.getenv("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings"),
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)
# 初始化远程 Embedding 模型（硅基流动）

if chunks:
    chunk_embeddings = embedding_model.encode(
        chunks,
        convert_to_tensor=True
    )
else:
    chunk_embeddings = []
# 把所有知识库资料提前转换成语义向量
# 这些资料没有变，所以程序启动时计算一次即可

