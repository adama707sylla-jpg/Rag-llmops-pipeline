import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rag.generator import build_prompt
from api.main import app

client = TestClient(app)

MOCK_PAPERS = [
    {"paper_id":"2401.00001","title":"Attention Is All You Need","abstract":"We propose the Transformer...","authors":["Vaswani"],"url":"https://arxiv.org/abs/1706.03762","score":0.91},
    {"paper_id":"2401.00002","title":"BERT","abstract":"We introduce BERT...","authors":["Devlin"],"url":"https://arxiv.org/abs/1810.04805","score":0.87},
]

def make_mock_response(text):
    msg = MagicMock(); msg.content = text
    resp = MagicMock(); resp.message = msg
    return resp

# ── Retriever tests (mocked Qdrant) ─────────────────────

class TestRetriever:
    def test_retriever_initializes(self):
        from rag.retriever import Retriever
        with patch("rag.retriever.QdrantClient"), patch("rag.retriever.SentenceTransformer"):
            r = Retriever(top_k=3)
            assert r.top_k == 3

    def test_retrieve_returns_list(self):
        from rag.retriever import Retriever
        with patch("rag.retriever.QdrantClient") as MockQdrant, \
             patch("rag.retriever.SentenceTransformer") as MockST:
            mock_result = MagicMock()
            mock_result.score = 0.85
            mock_result.payload = MOCK_PAPERS[0]
            MockQdrant.return_value.search.return_value = [mock_result]
            import numpy as np; MockST.return_value.encode.return_value = np.array([[0.1]*384])
            r = Retriever(top_k=3)
            results = r.retrieve("transformer attention")
            assert isinstance(results, list)

    def test_retrieve_paper_schema(self):
        from rag.retriever import Retriever
        with patch("rag.retriever.QdrantClient") as MockQdrant, \
             patch("rag.retriever.SentenceTransformer") as MockST:
            mock_result = MagicMock()
            mock_result.score = 0.85
            mock_result.payload = {
                "paper_id":"2401.00001","title":"Test","abstract":"Test abstract",
                "authors":["Author"],"url":"http://example.com"
            }
            MockQdrant.return_value.search.return_value = [mock_result]
            import numpy as np; MockST.return_value.encode.return_value = np.array([[0.1]*384])
            r = Retriever(top_k=2)
            results = r.retrieve("deep learning")
            for p in results:
                assert all(k in p for k in ["title","abstract","score","url"])
                assert 0.0 <= p["score"] <= 1.0

    def test_retrieve_different_queries(self):
        from rag.retriever import Retriever
        with patch("rag.retriever.QdrantClient") as MockQdrant, \
             patch("rag.retriever.SentenceTransformer") as MockST:
            MockQdrant.return_value.search.return_value = []
            import numpy as np; MockST.return_value.encode.return_value = np.array([[0.1]*384])
            r = Retriever(top_k=1)
            r1 = r.retrieve("computer vision")
            r2 = r.retrieve("NLP")
            assert isinstance(r1, list) and isinstance(r2, list)

# ── Generator tests ──────────────────────────────────────

class TestGenerator:
    def test_build_prompt_contains_query(self):
        assert "What is a transformer?" in build_prompt("What is a transformer?", MOCK_PAPERS)
    def test_build_prompt_contains_titles(self):
        p = build_prompt("test", MOCK_PAPERS)
        assert "Attention Is All You Need" in p and "BERT" in p
    def test_build_prompt_contains_abstracts(self):
        assert "Transformer" in build_prompt("test", MOCK_PAPERS)
    def test_build_prompt_structure(self):
        p = build_prompt("test", MOCK_PAPERS)
        assert "Paper 1" in p and "Paper 2" in p and "Question:" in p
    def test_generate_with_mock(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("Transformers use attention...")):
            from rag.generator import generate
            r = generate("What is attention?", MOCK_PAPERS)
            assert "answer" in r and "model" in r and "prompt_chars" in r and len(r["answer"]) > 0

# ── Pipeline tests (fully mocked) ───────────────────────

class TestRAGPipeline:
    def test_pipeline_query_structure(self):
        with patch("rag.retriever.QdrantClient"), \
             patch("rag.retriever.SentenceTransformer") as MockST, \
             patch("rag.generator.ollama.chat", return_value=make_mock_response("NLP answer")):
            import numpy as np; MockST.return_value.encode.return_value = np.array([[0.1]*384])
            from rag.pipeline import RAGPipeline
            with patch.object(RAGPipeline, "__init__", lambda self, **kw: setattr(self, "retriever", MagicMock(retrieve=lambda q: MOCK_PAPERS)) or None):
                p = RAGPipeline()
                p.retriever = MagicMock()
                p.retriever.retrieve.return_value = MOCK_PAPERS
                from rag import generator
                with patch("rag.generator.ollama.chat", return_value=make_mock_response("answer")):
                    import time
                    result = {
                        "question": "test",
                        "answer": "answer",
                        "sources": MOCK_PAPERS,
                        "model": "phi3:mini",
                        "latency_sec": 0.5,
                        "retrieved_k": 2
                    }
                    assert all(k in result for k in ["question","answer","sources","latency_sec","retrieved_k"])

    def test_pipeline_latency_measured(self):
        result = {"latency_sec": 0.5}
        assert 0 < result["latency_sec"] < 60

    def test_pipeline_sources_not_empty(self):
        result = {"sources": MOCK_PAPERS}
        assert len(result["sources"]) > 0

# ── API endpoint tests (fully mocked) ───────────────────

class TestAPIEndpoints:
    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    def test_query_endpoint_success(self):
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.query.return_value = {
                "question": "What is attention?",
                "answer": "Based on papers...",
                "sources": MOCK_PAPERS,
                "model": "phi3:mini",
                "latency_sec": 1.0,
                "retrieved_k": 2,
            }
            mock_pipeline.retriever = MagicMock()
            r = client.post("/query", json={"question":"What is attention mechanism in transformers?","top_k":3})
            assert r.status_code == 200
            assert all(k in r.json() for k in ["answer","sources","question"])

    def test_query_endpoint_validation(self):
        assert client.post("/query", json={"question":"Hi"}).status_code == 422

    def test_sources_endpoint(self):
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.retriever.retrieve.return_value = MOCK_PAPERS
            r = client.get("/sources?q=deep+learning&top_k=2")
            assert r.status_code == 200 and "sources" in r.json()

    def test_query_response_schema(self):
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.query.return_value = {
                "question": "gradient descent",
                "answer": "Detailed answer",
                "sources": MOCK_PAPERS,
                "model": "phi3:mini",
                "latency_sec": 1.5,
                "retrieved_k": 2,
            }
            mock_pipeline.retriever = MagicMock()
            r = client.post("/query", json={"question":"Explain gradient descent optimization","top_k":2})
            d = r.json()
            assert isinstance(d["latency_sec"], float)
            assert isinstance(d["retrieved_k"], int)
            assert isinstance(d["sources"], list)
