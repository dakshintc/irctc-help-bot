.PHONY: install ingest run offline test clean

install:
	pip install -r requirements.txt

ingest:
	python ingest.py

run:
	python app.py

offline:
	RAG_OFFLINE=1 python app.py

test:
	pytest

clean:
	rm -rf data/index data/chroma data/raw/*.txt data/raw/manifest.json data/build_info.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
