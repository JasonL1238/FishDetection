.PHONY: setup demo list-input

setup:
	python -m venv .venv
	.venv/bin/pip install -e .

demo:
	python -m fishdetect.cli hello

list-input:
	python -m fishdetect.cli list-input
