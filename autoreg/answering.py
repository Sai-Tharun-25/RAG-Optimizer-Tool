# autoreg/answering.py
from typing import List
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

@dataclass
class RAGSampleInput:
    query: str
    retrieved_docs: List[str]

class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        # Using flan-t5-base on CPU. First run downloads the model.
        self.model_name = model_name
        # initialize pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model_name,
            device=-1,  # CPU
            truncation=True
        )

    def build_prompt(self, sample: RAGSampleInput) -> str:
        # Label each retrieved chunk to make it clear for the model
        ctx = []
        for i, doc in enumerate(sample.retrieved_docs[:5]):  # use up to 5 chunks
            ctx.append(f"[DOC {i+1}]: {doc}")

        context_text = "\n\n".join(ctx)
        prompt = (
            f"You are given the following context documents. Use them to answer the question concisely "
            f"and in a complete sentence. If the context is insufficient, say 'Not enough information.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {sample.query}\n\nAnswer:"
        )
        return prompt

    def generate_answer(self, sample: RAGSampleInput) -> str:
        prompt = self.build_prompt(sample)

        out = self.generator(
            prompt,
            max_length=128,
            do_sample=False,   # deterministic
            num_return_sequences=1
        )
        text = out[0]["generated_text"].strip()

        # fallback: if model output is too short, return concatenated top-k chunks
        if len(text.split()) < 3:
            fallback = " ".join(sample.retrieved_docs[:3])
            return fallback[:1000].strip()

        return text
