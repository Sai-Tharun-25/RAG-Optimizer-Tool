from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.optimizer import grid_search

# --- RAG runtime imports (used by /query) ---
from autoreg.answering import AnswerGenerator, RAGSampleInput
from autoreg.chunking import build_chunk_corpus
from autoreg.retriever import SimpleEmbeddingModel, VectorRetriever

# Optional reranker (only if you have it)
try:
    from autoreg.reranker import CrossEncoderReranker
except Exception:
    CrossEncoderReranker = None


app = FastAPI(title="RAG Optimizer API")

# -----------------------------
# Existing Experiment Endpoint
# -----------------------------

class SearchSpace(BaseModel):
    chunk_size: List[int]
    overlap: List[int]
    k: List[int]

class ExperimentRequest(BaseModel):
    dataset_path: str
    docs_path: str
    search_space: SearchSpace

@app.post("/experiments")
def run_experiment(req: ExperimentRequest):
    dataset = load_qa_dataset(req.dataset_path)
    docs_corpus = load_docs_corpus(req.docs_path)

    results = grid_search(dataset, docs_corpus, req.search_space.dict())

    resp: List[Dict[str, Any]] = []
    for r in results:
        resp.append({
            "config": {
                "chunk_size": r.config.chunk_size,
                "overlap": r.config.overlap,
                "k": r.config.k,
            },
            "metrics": {
                "avg_f1_answer": r.avg_f1_answer,
                "avg_recall_at_k": r.avg_recall_at_k,
                "avg_judge_score": r.avg_judge_score,
            },
        })

    return {"results": resp}

# -----------------------------
# New Live RAG Query Endpoint
# -----------------------------

DOCS_PATH = Path("data/wiki_docs.jsonl")

DEFAULT_CHUNK_SIZE = 256
DEFAULT_OVERLAP = 64
DEFAULT_TOP_N = 20
DEFAULT_USE_RERANKER = True

# Globals initialized at startup
_docs: List[str] = []
_chunk_texts: List[str] = []
_chunk_doc_ids: List[int] = []
_retriever: Optional[VectorRetriever] = None
_reranker = None
_answer_gen: Optional[AnswerGenerator] = None


def _load_docs_jsonl(path: Path) -> List[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(obj["text"])
    return docs


@app.on_event("startup")
def startup_init():
    """
    Build everything once so /query is fast.
    """
    global _docs, _chunk_texts, _chunk_doc_ids, _retriever, _reranker, _answer_gen

    _docs = _load_docs_jsonl(DOCS_PATH)

    _chunk_texts, _chunk_doc_ids = build_chunk_corpus(
        _docs, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )

    embed_model = SimpleEmbeddingModel()
    _retriever = VectorRetriever(_chunk_texts, embed_model)

    if DEFAULT_USE_RERANKER and CrossEncoderReranker is not None:
        _reranker = CrossEncoderReranker()
    else:
        _reranker = None

    _answer_gen = AnswerGenerator(max_new_tokens=32, device=-1)


@app.get("/query")
def query_rag(
    q: str = Query(..., description="User question"),
    k: int = Query(3, ge=1, le=8, description="How many contexts to use"),
    use_reranker: bool = Query(DEFAULT_USE_RERANKER, description="Enable reranker"),
    top_n: int = Query(DEFAULT_TOP_N, ge=1, le=100, description="Candidates before rerank"),
):
    """
    Returns the exact shape the Streamlit UI expects:
    {
      query, answer,
      retrieved_chunks, retrieved_doc_ids, reranker_scores
    }
    """
    if _retriever is None or _answer_gen is None:
        return {"detail": "Server not initialized yet"}

    # 1) Retrieve top_n
    top_n = max(top_n, k)
    retrieved_topn = _retriever.retrieve(q, k=top_n)
    retrieved_topn_indices = [idx for idx, _ in retrieved_topn]
    retrieved_topn_texts = [_chunk_texts[idx] for idx in retrieved_topn_indices]

    reranker_scores = None

    # 2) Optional rerank
    final_chunk_indices = retrieved_topn_indices[:k]
    if use_reranker and _reranker is not None and retrieved_topn_texts:
        ranked = _reranker.rerank(q, retrieved_topn_texts)  # [(cand_idx, score), ...]
        topk_idx_in_candidates = [i for i, _ in ranked[:k]]
        final_chunk_indices = [retrieved_topn_indices[i] for i in topk_idx_in_candidates]
        reranker_scores = [float(s) for _, s in ranked[:k]]

    # 3) Build contexts
    retrieved_chunks = [_chunk_texts[i] for i in final_chunk_indices]
    retrieved_doc_ids = [int(_chunk_doc_ids[i]) for i in final_chunk_indices]

    # 4) Generate
    sample = RAGSampleInput(query=q, retrieved_docs=retrieved_chunks)
    answer = _answer_gen.generate_answer(sample)

    return {
        "query": q,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_doc_ids": retrieved_doc_ids,
        "reranker_scores": reranker_scores,
    }
