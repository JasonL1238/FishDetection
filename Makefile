.PHONY: setup tracking bg-subtraction clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e .

tracking:
	python src/fishdetect/simple_tracking_test.py

bg-subtraction:
	python apply_bg_subtraction_30_frames.py

clean:
	rm -rf .venv
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
