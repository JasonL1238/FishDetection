# Image Preprocessing Module

This module provides image preprocessing functionality for fish detection, specifically designed to work with background subtracted images.

## Features

- **Gaussian Blur**: Smooths images to reduce noise while preserving important features
- **Sharpening**: Enhances edges and details to improve fish detection accuracy
- **Configurable Parameters**: Adjustable kernel sizes, sigma values, and sharpening strength
- **Batch Processing**: Process entire directories of images
- **Comparison Tools**: Create side-by-side comparisons to visualize preprocessing effects

## Usage

### Basic Usage

```python
from image_pre import ImagePreprocessor

# Create preprocessor with default settings
preprocessor = ImagePreprocessor()

# Process a single image
processed_image = preprocessor.preprocess_image(image)

# Process a single file
preprocessor.process_image_file('input.png', 'output.png')

# Process entire directory
preprocessor.process_directory('input_dir/', 'output_dir/')
```

### Command Line Usage

```bash
# Process all background subtracted images with default settings
python src/image_pre/preprocess_bg_subtracted.py

# Custom parameters
python src/image_pre/preprocess_bg_subtracted.py \
    --kernel-size 7 7 \
    --sigma 1.5 \
    --sharpen-strength 1.2

# Skip blur or sharpening
python src/image_pre/preprocess_bg_subtracted.py --no-blur
python src/image_pre/preprocess_bg_subtracted.py --no-sharpening
```

### Create Comparison Images

```bash
# Create side-by-side comparisons
python src/image_pre/compare_preprocessing.py --num-samples 10
```

## Parameters

### Gaussian Blur
- **kernel_size**: Tuple of (width, height) for the Gaussian kernel (default: (5, 5))
- **sigma**: Standard deviation for Gaussian blur (default: 1.0)

### Sharpening
- **sharpen_kernel_strength**: Strength of the sharpening kernel (default: 1.0)
- Uses a 3x3 kernel: `[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]`

## File Structure

```
src/image_pre/
├── __init__.py                    # Module initialization
├── preprocessor.py               # Main preprocessing class
├── preprocess_bg_subtracted.py   # Command-line script
├── compare_preprocessing.py      # Comparison visualization script
└── README.md                     # This documentation
```

## Output

- **Preprocessed Images**: Saved to `output/preprocessed_images/` by default
- **Comparison Images**: Saved to `output/preprocessing_comparison/` by default
- **Naming Convention**: `{original_name}_preprocessed.png`

## Examples

### Example 1: Process with Custom Parameters

```python
from image_pre import ImagePreprocessor

# Create preprocessor with custom settings
preprocessor = ImagePreprocessor(
    gaussian_kernel_size=(7, 7),
    gaussian_sigma=1.5,
    sharpen_kernel_strength=1.2
)

# Process directory
count = preprocessor.process_directory(
    'input/bg_subtracted_frames/',
    'output/preprocessed_images/',
    file_pattern='frame_*_bg_subtracted.png'
)
print(f"Processed {count} images")
```

### Example 2: Process Only with Sharpening

```python
from image_pre import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Process with only sharpening (no blur)
processed = preprocessor.preprocess_image(
    image, 
    apply_blur=False, 
    apply_sharpening=True
)
```

## Integration with Fish Detection Pipeline

The preprocessed images are designed to work seamlessly with the existing fish detection pipeline:

1. **Background Subtraction**: Apply background subtraction first
2. **Preprocessing**: Apply Gaussian blur and sharpening
3. **Edge Detection**: Use Canny edge detection on preprocessed images
4. **Fish Detection**: Apply fish detection algorithms

This preprocessing step helps improve the quality of edge detection and subsequent fish detection by:
- Reducing noise through Gaussian blur
- Enhancing important features through sharpening
- Improving contrast between fish and background
