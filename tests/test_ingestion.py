"""
Tests unitaires — Pipeline d'ingestion Semaine 1
Couvre : arXiv fetch · MinIO upload · Qdrant indexing · search
"""

import pytest
import arxiv
import json
import pandas
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from ingestion.arxiv_ingestion import (
    fetch_arxiv_papers,
    get_minio_client,
    get_qdrant_client,
    upload_to_blob,
    index_in_vector_db,
    validate_pipeline,
    BUCKET_NAME,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def embed_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@pytest.fixture(scope="session")
def sample_papers():
    """Fetch a small batch — réutilisé par tous les tests."""
    return fetch_arxiv_papers(max_results=10)

@pytest.fixture(scope="session")
def minio():
    return get_minio_client()

@pytest.fixture(scope="session")
def qdrant():
    return get_qdrant_client()

# ── Tests ─────────────────────────────────────────────────

class TestArxivFetch:
    def test_returns_correct_count(self, sample_papers):
        assert len(sample_papers) == 10

    def test_paper_schema(self, sample_papers):
        required_keys = {"id", "title", "abstract", "authors", "categories", "published", "url", "text"}
        for paper in sample_papers:
            assert required_keys.issubset(paper.keys()), f"Missing keys in paper: {paper['id']}"

    def test_text_field_not_empty(self, sample_papers):
        for paper in sample_papers:
            assert len(paper["text"]) > 50, f"Text too short for paper {paper['id']}"

    def test_published_date_format(self, sample_papers):
        from datetime import datetime
        for paper in sample_papers:
            # Should be ISO format — parseable
            datetime.fromisoformat(paper["published"])


class TestMinIOUpload:
    def test_bucket_exists_after_upload(self, minio, sample_papers):
        upload_to_blob(minio, sample_papers)
        buckets = [b["Name"] for b in minio.list_buckets().get("Buckets", [])]
        assert BUCKET_NAME in buckets

    def test_objects_uploaded(self, minio, sample_papers):
        response = minio.list_objects_v2(Bucket=BUCKET_NAME, Prefix="raw/")
        count = response.get("KeyCount", 0)
        assert count >= len(sample_papers), f"Expected >= {len(sample_papers)}, got {count}"

    def test_object_content_valid_json(self, minio, sample_papers):
        paper_id = sample_papers[0]["id"]
        obj = minio.get_object(Bucket=BUCKET_NAME, Key=f"raw/{paper_id}.json")
        content = json.loads(obj["Body"].read())
        assert content["id"] == paper_id
        assert "title" in content
        assert "abstract" in content


class TestQdrantIndexing:
    def test_collection_exists(self, qdrant, sample_papers, embed_model):
        index_in_vector_db(qdrant, sample_papers, embed_model)
        collections = [c.name for c in qdrant.get_collections().collections]
        assert COLLECTION_NAME in collections

    def test_vectors_count(self, qdrant):
        info = qdrant.get_collection(COLLECTION_NAME)
        assert info.points_count >= 10

    def test_vector_dimension(self, qdrant):
        info = qdrant.get_collection(COLLECTION_NAME)
        assert info.config.params.vectors.size == 384


class TestVectorSearch:
    def test_search_returns_results(self, qdrant, embed_model):
        query_vec = embed_model.encode(["deep learning optimization"]).tolist()[0]
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=3,
        )
        assert len(results) == 3

    def test_search_score_range(self, qdrant, embed_model):
        query_vec = embed_model.encode(["neural network training"]).tolist()[0]
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=5,
        )
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score hors range: {r.score}"

    def test_search_payload_complete(self, qdrant, embed_model):
        query_vec = embed_model.encode(["transformer model"]).tolist()[0]
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=1,
        )
        payload = results[0].payload
        required = {"paper_id", "title", "abstract", "authors", "categories", "url"}
        assert required.issubset(payload.keys())

    def test_different_queries_give_different_results(self, qdrant, embed_model):
        q1 = embed_model.encode(["computer vision image classification"]).tolist()[0]
        q2 = embed_model.encode(["reinforcement learning reward"]).tolist()[0]
        r1 = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q1, limit=1)
        r2 = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q2, limit=1)
        assert r1[0].id != r2[0].id or r1[0].score != r2[0].score
