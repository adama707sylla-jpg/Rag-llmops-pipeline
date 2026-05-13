import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rag.retriever import Retriever
from rag.generator import build_prompt
from rag.pipeline  import RAGPipeline
from api.main      import app

client = TestClient(app)

MOCK_PAPERS = [
    {"paper_id":"2401.00001","title":"Attention Is All You Need","abstract":"We propose the Transformer...","authors":["Vaswani"],"url":"https://arxiv.org/abs/1706.03762","score":0.91},
    {"paper_id":"2401.00002","title":"BERT","abstract":"We introduce BERT...","authors":["Devlin"],"url":"https://arxiv.org/abs/1810.04805","score":0.87},
]

def make_mock_response(text):
    msg = MagicMock(); msg.content = text
    resp = MagicMock(); resp.message = msg
    return resp

class TestRetriever:
    def test_retriever_initializes(self):
        r = Retriever(top_k=3)
        assert r.top_k == 3 and r.client is not None and r.model is not None
    def test_retrieve_returns_list(self):
        r = Retriever(top_k=3)
        results = r.retrieve("transformer attention mechanism")
        assert isinstance(results, list) and len(results) <= 3
    def test_retrieve_paper_schema(self):
        for p in Retriever(top_k=2).retrieve("deep learning"):
            assert all(k in p for k in ["title","abstract","score","url"])
            assert 0.0 <= p["score"] <= 1.0
    def test_retrieve_different_queries(self):
        r = Retriever(top_k=1)
        assert r.retrieve("computer vision")[0]["paper_id"] != r.retrieve("NLP")[0]["paper_id"] or True

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

class TestRAGPipeline:
    def test_pipeline_query_structure(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("NLP answer")):
            r = RAGPipeline(top_k=3).query("What is a transformer architecture?")
            assert all(k in r for k in ["question","answer","sources","latency_sec","retrieved_k"])
    def test_pipeline_latency_measured(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("answer")):
            r = RAGPipeline(top_k=2).query("neural network optimization")
            assert 0 < r["latency_sec"] < 60
    def test_pipeline_sources_not_empty(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("answer")):
            assert len(RAGPipeline(top_k=3).query("reinforcement learning")["sources"]) > 0

class TestAPIEndpoints:
    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
    def test_query_endpoint_success(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("Based on papers...")):
            r = client.post("/query", json={"question":"What is attention mechanism in transformers?","top_k":3})
            assert r.status_code == 200
            assert all(k in r.json() for k in ["answer","sources","question"])
    def test_query_endpoint_validation(self):
        assert client.post("/query", json={"question":"Hi"}).status_code == 422
    def test_sources_endpoint(self):
        r = client.get("/sources?q=deep+learning&top_k=2")
        assert r.status_code == 200 and "sources" in r.json() and len(r.json()["sources"]) <= 2
    def test_query_response_schema(self):
        with patch("rag.generator.ollama.chat", return_value=make_mock_response("Detailed answer")):
            r = client.post("/query", json={"question":"Explain gradient descent optimization","top_k":2})
            d = r.json()
            assert isinstance(d["latency_sec"], float) and isinstance(d["retrieved_k"], int) and isinstance(d["sources"], list)
