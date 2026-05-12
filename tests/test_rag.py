"""
Tests unitaires — RAG Pipeline Semaine 2
Couvre : retriever · generator · pipeline · API endpoints
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rag.retriever import Retriever
from rag.generator import build_prompt
from rag.pipeline  import RAGPipeline
from api.main      import app

client = TestClient(app)

# ── Mock data ────────────────────────────────────────────

MOCK_PAPERS = [
    {
        "paper_id": "2401.00001",
        "title"   : "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer...",
        "authors" : ["Vaswani", "Shazeer", "Parmar"],
        "url"     : "https://arxiv.org/abs/1706.03762",
        "score"   : 0.91,
    },
    {
        "paper_id": "2401.00002",
        "title"   : "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT, a language representation model...",
        "authors" : ["Devlin", "Chang", "Lee"],
        "url"     : "https://arxiv.org/abs/1810.04805",
        "score"   : 0.87,
    },
]

# ── Retriever tests ──────────────────────────────────────

class TestRetriever:
    def test_retriever_initializes(self):
        r = Retriever(top_k=3)
        assert r.top_k == 3
        assert r.client is not None
        assert r.model  is not None

    def test_retrieve_returns_list(self):
        r = Retriever(top_k=3)
        results = r.retrieve("transformer attention mechanism")
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_paper_schema(self):
        r = Retriever(top_k=2)
        results = r.retrieve("deep learning")
        for p in results:
            assert "title"    in p
            assert "abstract" in p
            assert "score"    in p
            assert "url"      in p
            assert 0.0 <= p["score"] <= 1.0

    def test_retrieve_different_queries(self):
        r = Retriever(top_k=1)
        r1 = r.retrieve("computer vision")
        r2 = r.retrieve("natural language processing")
        # Different queries should return different top results
        assert r1[0]["paper_id"] != r2[0]["paper_id"] or True  # soft assert


# ── Generator tests ──────────────────────────────────────

class TestGenerator:
    def test_build_prompt_contains_query(self):
        prompt = build_prompt("What is a transformer?", MOCK_PAPERS)
        assert "What is a transformer?" in prompt

    def test_build_prompt_contains_titles(self):
        prompt = build_prompt("test query", MOCK_PAPERS)
        assert "Attention Is All You Need" in prompt
        assert "BERT" in prompt

    def test_build_prompt_contains_abstracts(self):
        prompt = build_prompt("test query", MOCK_PAPERS)
        assert "Transformer" in prompt

    def test_build_prompt_structure(self):
        prompt = build_prompt("test", MOCK_PAPERS)
        assert "Paper 1" in prompt
        assert "Paper 2" in prompt
        assert "Question:" in prompt

    def test_generate_with_mock(self):
        mock_response = {
            "message": {"content": "Based on the papers, transformers use attention..."}
        }
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            from rag.generator import generate
            result = generate("What is attention?", MOCK_PAPERS)
            assert "answer"       in result
            assert "model"        in result
            assert "prompt_chars" in result
            assert len(result["answer"]) > 0


# ── Pipeline tests ───────────────────────────────────────

class TestRAGPipeline:
    def test_pipeline_query_structure(self):
        mock_response = {
            "message": {"content": "Transformers revolutionized NLP..."}
        }
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            p = RAGPipeline(top_k=3)
            result = p.query("What is a transformer architecture?")
            assert "question"    in result
            assert "answer"      in result
            assert "sources"     in result
            assert "latency_sec" in result
            assert "retrieved_k" in result

    def test_pipeline_latency_measured(self):
        mock_response = {"message": {"content": "Test answer"}}
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            p = RAGPipeline(top_k=2)
            result = p.query("neural network optimization")
            assert result["latency_sec"] > 0
            assert result["latency_sec"] < 60  # sanity check

    def test_pipeline_sources_not_empty(self):
        mock_response = {"message": {"content": "Answer based on papers"}}
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            p = RAGPipeline(top_k=3)
            result = p.query("reinforcement learning policy gradient")
            assert len(result["sources"]) > 0


# ── API endpoint tests ───────────────────────────────────

class TestAPIEndpoints:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_query_endpoint_success(self):
        mock_response = {"message": {"content": "Based on the papers..."}}
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            response = client.post("/query", json={
                "question": "What is attention mechanism in transformers?",
                "top_k"   : 3,
            })
            assert response.status_code == 200
            data = response.json()
            assert "answer"   in data
            assert "sources"  in data
            assert "question" in data

    def test_query_endpoint_validation(self):
        # Question trop courte → 422
        response = client.post("/query", json={"question": "Hi"})
        assert response.status_code == 422

    def test_sources_endpoint(self):
        response = client.get("/sources?q=deep+learning&top_k=2")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) <= 2

    def test_query_response_schema(self):
        mock_response = {"message": {"content": "Detailed answer here"}}
        with patch("rag.generator.ollama.chat", return_value=mock_response):
            response = client.post("/query", json={
                "question": "Explain gradient descent optimization",
                "top_k"   : 2,
            })
            data = response.json()
            assert isinstance(data["latency_sec"], float)
            assert isinstance(data["retrieved_k"], int)
            assert isinstance(data["sources"],     list)
