"""
Retriever : query → embedding → Qdrant search → top-k papers
Equivalent de : Azure AI Search retrieval
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "arxiv-ml")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k  = top_k
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.model  = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Retriever ready — collection={COLLECTION_NAME}, top_k={top_k}")

    def retrieve(self, query: str) -> list[dict]:
        """Encode query → search Qdrant → return top-k papers."""
        query_vec = self.model.encode([query])[0].tolist()

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=self.top_k,
        )

        papers = []
        for r in results:
            papers.append({
                "paper_id" : r.payload.get("paper_id"),
                "title"    : r.payload.get("title"),
                "abstract" : r.payload.get("abstract"),
                "authors"  : r.payload.get("authors", []),
                "url"      : r.payload.get("url"),
                "score"    : round(r.score, 4),
            })
        return papers
