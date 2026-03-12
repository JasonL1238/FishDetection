# Fish Detection

Grid-based fish detection and tracking from video. Divides each frame into a 7x4 grid (28 cells) and detects one fish per cell using background subtraction and HSV masking.

## Setup

```bash
pip install -e .
```

## Usage

Place video files in `data/input/videos/`, then:

```bash
# Run the detection pipeline
python run.py

# Analyze per-fish tracking accuracy (after running the pipeline)
python analyze.py
```

## How it works

1. **Background model** -- median of sampled frames across the video
2. **Background subtraction** -- per-frame difference from the model, thresholded and morphologically cleaned
3. **HSV masking** -- filters for bright objects (fish) in the subtracted frame
4. **Grid detection** -- frame is divided into 28 cells (7 columns x 4 rows); the largest blob in each cell is taken as the fish
5. **Output** -- annotated video, per-frame PNG snapshots, and a summary report

## Project structure

```
FishDetection/
├── run.py                  # Pipeline entry point
├── analyze.py              # Per-fish tracking analysis
├── pyproject.toml          # Package config and dependencies
├── data/
│   └── input/
│       └── videos/         # Place input videos here (gitignored)
└── src/
    └── fishdetection/      # Core library
        ├── __init__.py
        ├── background_subtractor.py
        ├── base_masker.py
        ├── hsv_masker.py
        ├── base_pipeline.py
        ├── grid.py
        └── pipeline.py
```
