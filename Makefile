.PHONY: up down build test logs clean api

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build api

test:
	PYTHONUTF8=1 pytest tests/ -v --tb=short

test-unit:
	PYTHONUTF8=1 pytest tests/test_rag.py -v --tb=short

test-ingestion:
	PYTHONUTF8=1 pytest tests/test_ingestion.py -v --tb=short

api:
	PYTHONUTF8=1 uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

logs:
	docker-compose logs -f api

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
