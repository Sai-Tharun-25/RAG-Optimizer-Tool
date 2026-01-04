from dataclasses import dataclass
from typing import List

from .data import QAExample
from .retriever import VectorRetriever, SimpleEmbeddingModel
from .metrics import recall_at_k, f1_token_overlap
from .answering import AnswerGenerator, RAGSampleInput
from .chunking import build_chunk_corpus
from .judge import LLMJudge, JudgeInput


@dataclass
class RAGConfig:
    chunk_size: int
    overlap: int
    k: int


@dataclass
class RAGResult:
    config: RAGConfig
    avg_recall_at_k: float
    avg_f1_answer: float
    avg_judge_score: float


def run_rag_experiment(
    dataset: List[QAExample],
    config: RAGConfig,
    docs_corpus: List[str],
) -> RAGResult:
    """
    Runs one RAG experiment given a dataset, config, and raw docs.
    Returns average retrieval recall, answer F1 (lexical), and judge score (semantic).
    """

    # Build chunk-level corpus according to config
    chunk_texts, chunk_doc_ids = build_chunk_corpus(
        docs_corpus, chunk_size=config.chunk_size, overlap=config.overlap
    )

    # Shared embedding model (singleton) and retriever built on chunk_texts
    embed_model = SimpleEmbeddingModel()
    retriever = VectorRetriever(chunk_texts, embed_model)

    # Answer generator (LLM or local model)
    answer_gen = AnswerGenerator()

    # LLM-based judge (semantic scorer)
    judge = LLMJudge()

    recalls = []
    f1_scores = []
    judge_scores = []

    for ex in dataset:
        # Retrieve top-k chunk indices & texts
        retrieved = retriever.retrieve(ex.query, k=config.k)
        retrieved_chunk_indices = [idx for idx, _ in retrieved]
        retrieved_chunk_texts = [chunk_texts[idx] for idx in retrieved_chunk_indices]

        # Map chunk indices -> original doc ids for recall computation
        retrieved_doc_ids = {chunk_doc_ids[idx] for idx in retrieved_chunk_indices}

        # Retrieval metric (Recall@k)
        relevant_indices = set(ex.relevant_doc_ids or [])
        recall_val = recall_at_k(list(retrieved_doc_ids), relevant_indices)
        recalls.append(recall_val)

        # Answer generation (use top retrieved chunks as context)
        sample = RAGSampleInput(query=ex.query, retrieved_docs=retrieved_chunk_texts)
        pred_answer = answer_gen.generate_answer(sample)

        # Lexical F1 (fallback metric)
        f1 = f1_token_overlap(pred_answer, ex.answer)
        f1_scores.append(f1)

        # LLM-based judge scoring (semantic, 0.0-1.0)
        context_concat = "\n\n".join(retrieved_chunk_texts[:5])  # top-5 chunks for judge
        judge_in = JudgeInput(
            query=ex.query,
            context=context_concat,
            predicted=pred_answer,
            gold=ex.answer,
        )
        try:
            jscore = judge.score(judge_in)
        except Exception:
            # if judge fails for any example, fallback to lexical F1 normalized
            jscore = max(0.0, min(1.0, f1))
        judge_scores.append(jscore)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    return RAGResult(
        config=config,
        avg_recall_at_k=avg_recall,
        avg_f1_answer=avg_f1,
        avg_judge_score=avg_judge,
    )
