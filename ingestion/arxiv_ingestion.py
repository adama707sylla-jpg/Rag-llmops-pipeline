"""
Pipeline : arXiv API → MinIO (Blob) → Qdrant (Vector DB)
Equivalent local de : Azure Blob Storage + Azure AI Search
"""

import os
import json
import arxiv
import boto3
from tqdm import tqdm
from dotenv import load_dotenv
from botocore.client import Config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from datetime import datetime

load_dotenv()

# ── Config ──────────────────────────────────────────────
STORAGE_ENDPOINT   = os.getenv("STORAGE_ENDPOINT", "http://localhost:9000")
STORAGE_ACCESS_KEY = os.getenv("STORAGE_ACCESS_KEY", "minioadmin")
STORAGE_SECRET_KEY = os.getenv("STORAGE_SECRET_KEY", "minioadmin")
BUCKET_NAME        = os.getenv("STORAGE_BUCKET", "arxiv-papers")
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME    = os.getenv("QDRANT_COLLECTION", "arxiv-ml")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_PAPERS         = 500

# ── Clients ─────────────────────────────────────────────
def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=STORAGE_ENDPOINT,
        aws_access_key_id=STORAGE_ACCESS_KEY,
        aws_secret_access_key=STORAGE_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ── Step 1 : Fetch arXiv papers ─────────────────────────
def fetch_arxiv_papers(max_results: int = MAX_PAPERS) -> list[dict]:
    print(f"📥 Fetching {max_results} papers from arXiv...")
    search = arxiv.Search(
        query="machine learning OR deep learning OR MLOps OR LLM",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in tqdm(search.results(), total=max_results):
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary.replace("\n", " "),
            "authors": [a.name for a in result.authors[:5]],
            "categories": result.categories,
            "published": result.published.isoformat(),
            "url": result.entry_id,
            "text": f"Title: {result.title}\n\nAbstract: {result.summary}",
        })
    print(f"✅ Fetched {len(papers)} papers")
    return papers

# ── Step 2 : Upload to MinIO (= Azure Blob Storage) ─────
def upload_to_blob(client, papers: list[dict]):
    print(f"\n📦 Uploading to MinIO (Azure Blob equivalent)...")

    # Create bucket if not exists
    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if BUCKET_NAME not in existing:
        client.create_bucket(Bucket=BUCKET_NAME)
        print(f"  Bucket '{BUCKET_NAME}' created")

    for paper in tqdm(papers):
        key = f"raw/{paper['id']}.json"
        client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=json.dumps(paper, ensure_ascii=False),
            ContentType="application/json",
        )
    print(f"✅ {len(papers)} papers uploaded to MinIO/{BUCKET_NAME}")

# ── Step 3 : Embed + Index in Qdrant (= Azure AI Search) 
def index_in_vector_db(qdrant: QdrantClient, papers: list[dict], model: SentenceTransformer):
    print(f"\n🔢 Generating embeddings with {EMBEDDING_MODEL}...")

    # Create collection if not exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"  Collection '{COLLECTION_NAME}' created")

    texts = [p["text"] for p in papers]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    print(f"\n📌 Indexing {len(papers)} vectors in Qdrant...")
    points = [
        PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={
                "paper_id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "authors": p["authors"],
                "categories": p["categories"],
                "published": p["published"],
                "url": p["url"],
            },
        )
        for idx, p in enumerate(papers)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ {len(points)} vectors indexed in Qdrant/{COLLECTION_NAME}")

# ── Step 4 : Validation ─────────────────────────────────
def validate_pipeline(minio, qdrant):
    print("\n🔍 Validating pipeline...")

    # Check MinIO
    objects = minio.list_objects_v2(Bucket=BUCKET_NAME, Prefix="raw/")
    count_blob = objects.get("KeyCount", 0)
    print(f"  MinIO  : {count_blob} objects in bucket")

    # Check Qdrant
    info = qdrant.get_collection(COLLECTION_NAME)
    count_vec = info.points_count
    print(f"  Qdrant : {count_vec} vectors indexed")

    # Test search
    test_query = "transformer architecture attention mechanism"
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([test_query])[0].tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=3,
    )
    print(f"\n  Test query : '{test_query}'")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r.score:.3f} | {r.payload['title'][:70]}...")

    print("\n✅ Pipeline validated successfully!")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG LLMOps — Ingestion Pipeline (Week 1)")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    minio_client  = get_minio_client()
    qdrant_client = get_qdrant_client()
    embed_model   = SentenceTransformer(EMBEDDING_MODEL)

    papers = fetch_arxiv_papers(MAX_PAPERS)
    upload_to_blob(minio_client, papers)
    index_in_vector_db(qdrant_client, papers, embed_model)
    validate_pipeline(minio_client, qdrant_client)
