# fishdetect

A computer vision package for detecting tiny black fish in images, with filtering to exclude bars and other dark objects.

## Features

- **Tiny Fish Detection**: Specifically designed to detect small black fish
- **Smart Filtering**: Excludes bars, lines, and other non-fish dark objects
- **Configurable Parameters**: Adjustable size, shape, and color thresholds
- **Batch Processing**: Process multiple images at once
- **Visualization**: Generate annotated images showing detected fish

## Quick Start

```bash
# Install dependencies
make setup

# Run detection on an image
python -m fishdetect.cli detect input/your_image.png --details

# Or use the simple demo script
python demo.py input/your_image.png
```

## Usage

### Single Image Detection

```bash
# Basic detection
python -m fishdetect.cli detect input/image.png

# With detailed output and custom parameters
python -m fishdetect.cli detect input/image.png \
  --min-area 100 \
  --max-area 1500 \
  --min-aspect 0.4 \
  --max-aspect 2.5 \
  --black-threshold 40 \
  --details
```

### Batch Processing

```bash
# Process all images in input directory
python -m fishdetect.cli detect-all

# With custom output directory
python -m fishdetect.cli detect-all --output results/
```

### Parameters

- `--min-area`: Minimum area for fish detection (default: 50)
- `--max-area`: Maximum area for fish detection (default: 2000)
- `--min-aspect`: Minimum width/height ratio (default: 0.3)
- `--max-aspect`: Maximum width/height ratio (default: 3.0)
- `--black-threshold`: Threshold for black color detection (default: 50)

## Algorithm

The detection algorithm uses:

1. **Preprocessing**: Gaussian blur and adaptive thresholding
2. **Contour Detection**: Find all dark objects in the image
3. **Shape Filtering**: Filter by area, aspect ratio, and circularity
4. **Color Validation**: Verify objects are actually black
5. **Non-Maximum Suppression**: Remove overlapping detections

## Examples

- Add your images to the `input/` folder
- List input frames: `make list-input`
- Run CLI commands: `python -m fishdetect.cli --help`
