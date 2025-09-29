#!/usr/bin/env python3
"""
Script to create comparison images showing the effects of preprocessing.

This script creates side-by-side comparisons of original, background subtracted,
and preprocessed images to visualize the preprocessing effects.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from image_pre import ImagePreprocessor


def create_comparison_image(original_path, bg_subtracted_path, preprocessed_path, output_path):
    """
    Create a side-by-side comparison of original, bg subtracted, and preprocessed images.
    """
    # Read images
    original = cv2.imread(str(original_path))
    bg_subtracted = cv2.imread(str(bg_subtracted_path))
    preprocessed = cv2.imread(str(preprocessed_path))
    
    if original is None or bg_subtracted is None or preprocessed is None:
        print(f"Error: Could not read one or more images")
        return False
    
    # Resize images to same height (keep aspect ratio)
    target_height = 300
    h, w = original.shape[:2]
    new_width = int(w * target_height / h)
    
    original_resized = cv2.resize(original, (new_width, target_height))
    bg_subtracted_resized = cv2.resize(bg_subtracted, (new_width, target_height))
    preprocessed_resized = cv2.resize(preprocessed, (new_width, target_height))
    
    # Create comparison image
    comparison = np.hstack([original_resized, bg_subtracted_resized, preprocessed_resized])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "Background Subtracted", (new_width + 10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "Preprocessed", (2 * new_width + 10, 30), font, font_scale, color, thickness)
    
    # Save comparison
    cv2.imwrite(str(output_path), comparison)
    return True


def main():
    parser = argparse.ArgumentParser(description='Create preprocessing comparison images')
    parser.add_argument('--input-dir', 
                       default='input/bg_subtracted_frames',
                       help='Input directory containing original and bg subtracted images')
    parser.add_argument('--preprocessed-dir',
                       default='output/preprocessed_images',
                       help='Directory containing preprocessed images')
    parser.add_argument('--output-dir',
                       default='output/preprocessing_comparison',
                       help='Output directory for comparison images')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of sample images to create comparisons for')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    preprocessed_path = Path(args.preprocessed_dir)
    output_path = Path(args.output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find available images
    original_files = list(input_path.glob("frame_*_original.png"))
    original_files.sort()
    
    if not original_files:
        print(f"No original images found in {input_path}")
        return
    
    # Process sample images
    num_samples = min(args.num_samples, len(original_files))
    successful_count = 0
    
    for i in range(num_samples):
        original_file = original_files[i]
        frame_num = original_file.stem.replace('_original', '')
        
        # Find corresponding files
        bg_subtracted_file = input_path / f"{frame_num}_bg_subtracted.png"
        preprocessed_file = preprocessed_path / f"{frame_num}_bg_subtracted_preprocessed.png"
        
        if not bg_subtracted_file.exists():
            print(f"Warning: {bg_subtracted_file} not found, skipping")
            continue
            
        if not preprocessed_file.exists():
            print(f"Warning: {preprocessed_file} not found, skipping")
            continue
        
        # Create comparison
        output_file = output_path / f"comparison_{frame_num}.png"
        
        if create_comparison_image(original_file, bg_subtracted_file, preprocessed_file, output_file):
            successful_count += 1
            print(f"Created comparison: {output_file.name}")
        else:
            print(f"Failed to create comparison for {frame_num}")
    
    print(f"\\nCreated {successful_count}/{num_samples} comparison images")
    print(f"Comparison images saved to: {output_path}")


if __name__ == "__main__":
    main()
