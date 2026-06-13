.PHONY: seed run dev test lint

seed:
	python -m engine.db.seed

run:
	uvicorn engine.main:app --port 8001 --reload

dev:
	cd frontend && npm run dev

install:
	pip install -r requirements.txt
	cd frontend && npm install

test:
	pytest tests/ -v

lint:
	ruff check engine/

fix:
	ruff check --fix engine/
