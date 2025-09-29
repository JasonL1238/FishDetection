#!/usr/bin/env python3
"""
Process preprocessed images with Canny edge detection.

This script applies Canny edge detection to the preprocessed images
with improved parameters for fish detection, including minimum area
filtering and red outline visualization.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from masking_methods.canny_edge_detection.canny_masker import CannyMasker


def main():
    """Process preprocessed images with Canny edge detection."""
    
    # Define paths
    input_dir = "output/preprocessed_images"
    output_dir = "output/canny_edges_preprocessed"
    comparison_dir = "output/canny_preprocessed_comparison"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize Canny masker with optimized parameters
    canny_masker = CannyMasker(
        low_threshold=50,
        high_threshold=150,
        aperture_size=3,
        l2_gradient=False,
        blur_kernel_size=3,  # Reduced blur for preprocessed images
        morphology_kernel_size=3
    )
    
    # Get all preprocessed images
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*_preprocessed.png"))
    image_files.sort()
    
    print(f"Found {len(image_files)} preprocessed images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing {image_file.name}...")
        
        # Load image
        frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        
        if frame is None:
            print(f"Warning: Could not load image {image_file}")
            continue
        
        # Process with Canny edge detection
        edges = canny_masker.process_frame(frame)
        
        # Create visualization with red outlines and area filtering
        visualization = canny_masker.visualize_detection(frame, min_area=500, max_area=10000)
        
        # Save edge detection result
        edge_filename = f"frame_{i:03d}_canny_edges.png"
        edge_path = Path(output_dir) / edge_filename
        cv2.imwrite(str(edge_path), edges)
        
        # Save visualization
        vis_filename = f"frame_{i:03d}_canny_visualization.png"
        vis_path = Path(output_dir) / vis_filename
        cv2.imwrite(str(vis_path), visualization)
        
        # Create comparison image (original + edges + visualization)
        if len(frame.shape) == 2:
            original_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = frame.copy()
        
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_bgr, edges_bgr, visualization])
        
        # Save comparison
        comparison_filename = f"comparison_frame_{i:03d}.png"
        comparison_path = Path(comparison_dir) / comparison_filename
        cv2.imwrite(str(comparison_path), comparison)
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print(f"\nProcessing complete!")
    print(f"Edge detection results saved to: {output_dir}")
    print(f"Comparison images saved to: {comparison_dir}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Processed {len(image_files)} preprocessed images")
    print(f"- Applied minimum area filter of 500 pixels to remove small spots")
    print(f"- Used red outlines for fish detection visualization")
    print(f"- Generated edge detection results and comparison images")


if __name__ == "__main__":
    main()
