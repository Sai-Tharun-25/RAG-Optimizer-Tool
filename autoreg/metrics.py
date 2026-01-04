from typing import List, Set
import re

def recall_at_k(retrieved_indices: List[int], relevant_indices: Set[int]) -> float:
    if not relevant_indices:
        return 0.0
    hit_count = sum(1 for i in retrieved_indices if i in relevant_indices)
    return hit_count / len(relevant_indices)

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\w+", text)

def f1_token_overlap(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    common = pred_set & gold_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)