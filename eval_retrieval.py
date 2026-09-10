from app.retrieval import (
    semantic_retrieve,
    gate,
    chunks,
    chunk_embeddings,
    embedding_model
)
cases = [
    {
        "name": "Case A",
        "question": "RAG 的核心思想是什么？",
        "expected_sources": [
            "RAG 是 Retrieval Augmented Generation 的缩写，核心思想是在生成回答之前先从外部知识库中检索相关资料。"
        ]
    },
    {
        "name": "Case B",
        "question": "怎样让模型在回答前先参考外部资料？",
        "expected_sources": [
            "RAG 是 Retrieval Augmented Generation 的缩写，核心思想是在生成回答之前先从外部知识库中检索相关资料。"
        ]
    },
    {
        "name": "Case C",
        "question": "GPT 是什么时候诞生的？",
        "expected_sources": []
    },
    {
        "name": "Case D",
        "question": "RAG 中 Retrieval 和 Generation 分别负责什么？",
        "expected_sources": [
            "Retrieval 是 RAG 中负责查找相关资料的阶段，它的输出可以作为后续大语言模型回答问题时的参考信息。",
            "Generation 是 RAG 中由大语言模型根据用户问题和提供的 Context 生成最终回答的阶段。"
        ]
    }
]
passed_count = 0
for case in cases:
    question = case["question"]
    expected_sources = case["expected_sources"]
    top_results = semantic_retrieve(
    question,
    chunks,
    chunk_embeddings,
    embedding_model,
    )


    best_score = top_results[0][0]

    matched = gate(best_score)

    if matched == False:
        actual_sources = []
    else:
        actual_sources = []

        for result in top_results:
            if result[0] >= 0.3:
                actual_sources.append(result[1])
            else:
                break



    if expected_sources == []:
        passed = actual_sources == []
    else:
        passed = True

        for expected_source in expected_sources:
            if expected_source not in actual_sources:
                passed = False
                break

    if passed:
        passed_count += 1
        result_text = "PASS"
    else:
        result_text = "FAIL"

    print(case["name"], "|", result_text)

print("Passed:", passed_count, "/", len(cases))