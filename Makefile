.PHONY: install lab test lint notebooks

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

lab:
	jupyter lab

test:
	python -m pytest

lint:
	ruff check src tests

notebooks:
	python scripts/validate_notebooks.py

