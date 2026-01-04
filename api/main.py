from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.optimizer import grid_search

app = FastAPI()

class SearchSpace(BaseModel):
    chunk_size: List[int]
    overlap: List[int]
    k: List[int]

class ExperimentRequest(BaseModel):
    dataset_path: str  # for now, path on server (e.g. "data/sample.jsonl")
    docs_path: str     # e.g. "data/docs.jsonl"
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