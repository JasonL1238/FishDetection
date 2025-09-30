#!/usr/bin/env python3
"""
Script to preprocess tracked frames with Gaussian blur and sharpening.

This script applies the preprocessing pipeline (blur + sharpen) to all tracked frames
and saves them to a new preprocessed_tracked_frames folder.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from image_pre import ImagePreprocessor


def main():
    # Define paths
    input_dir = Path("/Users/jasonli/FishDetection/data/input/tracked_frames")
    output_dir = Path("/Users/jasonli/FishDetection/data/input/frames/preprocessed_tracked_frames")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor with default settings
    preprocessor = ImagePreprocessor(
        gaussian_kernel_size=(5, 5),
        gaussian_sigma=1.0,
        sharpen_kernel_strength=1.0
    )
    
    # Get all tracked frame files
    tracked_files = list(input_dir.glob("frame_*_tracked.png"))
    tracked_files.sort()
    
    print(f"Found {len(tracked_files)} tracked frames to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    successful_count = 0
    failed_count = 0
    
    for i, input_file in enumerate(tracked_files):
        try:
            # Read the image
            image = cv2.imread(str(input_file))
            if image is None:
                print(f"Warning: Could not read {input_file.name}")
                failed_count += 1
                continue
            
            # Apply preprocessing (blur + sharpen)
            processed_image = preprocessor.preprocess_image(
                image, 
                apply_blur=True, 
                apply_sharpening=True
            )
            
            # Create output filename (replace _tracked with _preprocessed)
            output_filename = input_file.name.replace("_tracked.png", "_preprocessed.png")
            output_path = output_dir / output_filename
            
            # Save processed image
            success = cv2.imwrite(str(output_path), processed_image)
            if success:
                successful_count += 1
                if (i + 1) % 100 == 0:  # Progress update every 100 files
                    print(f"Processed {i + 1}/{len(tracked_files)} files...")
            else:
                print(f"Warning: Failed to save {output_filename}")
                failed_count += 1
                
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
            failed_count += 1
    
    print(f"\nPreprocessing completed!")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
