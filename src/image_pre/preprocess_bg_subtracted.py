#!/usr/bin/env python3
"""
Script to preprocess background subtracted images with Gaussian blur and sharpening.

Usage:
    python preprocess_bg_subtracted.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--no-blur] [--no-sharpening]
"""

import argparse
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from image_pre import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description='Preprocess background subtracted images')
    parser.add_argument('--input-dir', 
                       default='output/bg_subtracted_30_frames',
                       help='Input directory containing background subtracted images')
    parser.add_argument('--output-dir',
                       default='output/preprocessed_images',
                       help='Output directory for preprocessed images')
    parser.add_argument('--no-blur', action='store_true',
                       help='Skip Gaussian blur preprocessing')
    parser.add_argument('--no-sharpening', action='store_true',
                       help='Skip sharpening preprocessing')
    parser.add_argument('--kernel-size', type=int, nargs=2, default=[5, 5],
                       help='Gaussian kernel size (width height)')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Gaussian sigma value')
    parser.add_argument('--sharpen-strength', type=float, default=1.0,
                       help='Sharpening kernel strength')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        gaussian_kernel_size=tuple(args.kernel_size),
        gaussian_sigma=args.sigma,
        sharpen_kernel_strength=args.sharpen_strength
    )
    
    # Process images
    successful_count = preprocessor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern="frame_*_bg_subtracted.png",
        apply_blur=not args.no_blur,
        apply_sharpening=not args.no_sharpening
    )
    
    if successful_count > 0:
        print(f"\nPreprocessing completed successfully!")
        print(f"Processed {successful_count} images")
        print(f"Output saved to: {args.output_dir}")
    else:
        print("\nNo images were processed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
