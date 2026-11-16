import asyncio
import json
import math
import os
import sys
import unittest
from unittest.mock import patch

import requests


def _unit_vector_for_cosine(target_similarity: float) -> list[float]:
    """构造与 [1.0, 0.0] 余弦相似度恰好等于 target_similarity 的二维单位向量。"""
    y = math.sqrt(1.0 - target_similarity * target_similarity)
    return [target_similarity, y]


class _FakeEmbeddingResponse:
    """用于屏蔽模块导入阶段远程 Embedding 请求的最小响应替身。"""

    status_code = 200

    def __init__(self, count: int):
        self._count = count

    def json(self) -> dict:
        return {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in range(self._count)
            ]
        }


def _import_app_main_offline():
    """在不触发真实远程 Embedding 网络请求的前提下导入 app.main。"""
    if "app.main" in sys.modules:
        return sys.modules["app.main"]

    def fake_post(*args, **kwargs):
        del args
        inputs = kwargs.get("json", {}).get("input", [])
        return _FakeEmbeddingResponse(len(inputs))

    env_patch = {}
    if not os.getenv("EMBEDDING_API_KEY"):
        env_patch["EMBEDDING_API_KEY"] = "test-embedding-key"

    with patch.dict(os.environ, env_patch):
        with patch.object(requests, "post", side_effect=fake_post):
            import app.main as app_main

    return app_main


class _FakeEmbeddingModel:
    """返回固定查询向量 [1.0, 0.0]，使相似度完全由候选向量决定。"""

    def encode(self, texts, convert_to_tensor: bool = False):
        del texts, convert_to_tensor
        return [1.0, 0.0]


async def _post_ask_via_asgi(asgi_app, payload: dict) -> tuple[int, dict]:
    """直接通过 ASGI 调用 FastAPI /ask 接口，避免依赖额外测试客户端库。"""
    body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    request_sent = False
    status_code = None
    response_body_parts = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body_parts.append(message.get("body", b""))

    await asgi_app(scope, receive, send)
    raw_body = b"".join(response_body_parts).decode("utf-8")
    return status_code, json.loads(raw_body)


class AskSourceFilteringTest(unittest.TestCase):
    """验证 /ask 接口会过滤低于 0.3 阈值的检索资料，且返回 sources 与 Context 一致。"""

    @classmethod
    def setUpClass(cls):
        cls.app_main = _import_app_main_offline()

    def test_ask_excludes_sources_below_threshold_and_matches_context(self):
        chunk_a = "资料 A（高相关，0.85）：RAG 先检索资料再生成回答。"
        chunk_b = "资料 B（中相关，0.45）：Retrieval 负责查找参考文本。"
        chunk_c = "资料 C（低相关，0.20）：今天天气晴朗，适合出门散步。"

        # 刻意打乱知识库顺序，同时覆盖排序与 0.3 阈值过滤
        controlled_chunks = [chunk_c, chunk_a, chunk_b]
        controlled_embeddings = [
            _unit_vector_for_cosine(0.20),
            _unit_vector_for_cosine(0.85),
            _unit_vector_for_cosine(0.45),
        ]

        captured_calls = []

        def fake_ask_llm(prompt: str, model: str) -> str:
            captured_calls.append({"prompt": prompt, "model": model})
            return "模拟回答：已根据高相关参考资料作答。"

        with (
            patch.object(self.app_main, "chunks", controlled_chunks),
            patch.object(self.app_main, "chunk_embeddings", controlled_embeddings),
            patch.object(self.app_main, "embedding_model", _FakeEmbeddingModel()),
            patch.object(self.app_main, "ask_llm", side_effect=fake_ask_llm),
        ):
            status_code, response_data = asyncio.run(
                _post_ask_via_asgi(
                    self.app_main.app,
                    {"question": "什么是 RAG？", "top_k": 3},
                )
            )

        self.assertEqual(status_code, 200)
        self.assertEqual(response_data["sources"], [chunk_a, chunk_b])
        self.assertNotIn(chunk_c, response_data["sources"])
        self.assertAlmostEqual(response_data["best_score"], 0.85, places=6)
        self.assertEqual(response_data["answer"], "模拟回答：已根据高相关参考资料作答。")

        # 确认实际传给 LLM 的 Context 与响应中的 sources 完全一致，且不包含低分资料 C
        self.assertEqual(len(captured_calls), 1)
        captured_prompt = captured_calls[0]["prompt"]
        self.assertIn(chunk_a, captured_prompt)
        self.assertIn(chunk_b, captured_prompt)
        self.assertNotIn(chunk_c, captured_prompt)

        context_marker_start = "参考资料：\n        "
        context_marker_end = "\n\n        用户问题："
        start_idx = captured_prompt.index(context_marker_start) + len(context_marker_start)
        end_idx = captured_prompt.index(context_marker_end)
        actual_context_text = captured_prompt[start_idx:end_idx]
        self.assertEqual(actual_context_text, "\n\n".join(response_data["sources"]))


if __name__ == "__main__":
    unittest.main()
