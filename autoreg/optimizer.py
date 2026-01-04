from typing import List, Dict, Any
from .data import QAExample
from .rag_pipeline import RAGConfig, run_rag_experiment, RAGResult

def grid_search(
    dataset: List[QAExample],
    docs_corpus: List[str],
    search_space: Dict[str, List[Any]],
) -> List[RAGResult]:
    results: List[RAGResult] = []

    for chunk_size in search_space["chunk_size"]:
        for overlap in search_space["overlap"]:
            if overlap >= chunk_size:
                continue
            for k in search_space["k"]:
                config = RAGConfig(
                    chunk_size=chunk_size,
                    overlap=overlap,
                    k=k,
                )
                res = run_rag_experiment(dataset, config, docs_corpus)
                results.append(res)

    # Sort primarily by avg_judge_score, then by avg_f1_answer, then recall
    results.sort(
        key=lambda r: (r.avg_judge_score, r.avg_f1_answer, r.avg_recall_at_k),
        reverse=True
    )
    return results
