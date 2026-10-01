.PHONY: setup eval train test demo lint format clean

setup:
	pip install -e .[dev]

test:
	pytest tests/ -v --cov=src/expresseval

eval:
	expresseval run --manifest data/manifests/sample_manifest.csv --config configs/default.yaml

train:
	python -m expresseval.aggregator.train --config configs/train_aggregator.yaml

demo:
	python app/gradio_app.py

lint:
	ruff check src/ tests/

format:
	black src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
