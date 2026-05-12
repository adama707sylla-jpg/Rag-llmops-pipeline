"""
RAG Pipeline : orchestration retriever + generator
"""

import time
from rag.retriever import Retriever
from rag.generator import generate

class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)

    def query(self, question: str) -> dict:
        start = time.time()

        # Step 1 — Retrieve
        papers = self.retriever.retrieve(question)

        # Step 2 — Generate
        generation = generate(question, papers)

        elapsed = round(time.time() - start, 2)

        return {
            "question"      : question,
            "answer"        : generation["answer"],
            "sources"       : papers,
            "model"         : generation["model"],
            "latency_sec"   : elapsed,
            "retrieved_k"   : len(papers),
        }
