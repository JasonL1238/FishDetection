"""
Image preprocessing functions for fish detection.

This module provides preprocessing functions including Gaussian blur and sharpening
for background subtracted images to improve fish detection accuracy.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import os


class ImagePreprocessor:
    """
    Image preprocessing class for fish detection.
    
    Provides methods for Gaussian blur and sharpening preprocessing
    of background subtracted images.
    """
    
    def __init__(self, 
                 gaussian_kernel_size: Tuple[int, int] = (5, 5),
                 gaussian_sigma: float = 1.0,
                 sharpen_kernel_strength: float = 1.0):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            gaussian_kernel_size: Kernel size for Gaussian blur (width, height)
            gaussian_sigma: Standard deviation for Gaussian blur
            sharpen_kernel_strength: Strength of the sharpening kernel
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.sharpen_kernel_strength = sharpen_kernel_strength
        
        # Create sharpening kernel
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]]) * sharpen_kernel_strength
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Blurred image as numpy array
        """
        return cv2.GaussianBlur(image, self.gaussian_kernel_size, self.gaussian_sigma)
    
    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpened image as numpy array
        """
        # Apply the sharpening kernel using filter2D
        sharpened = cv2.filter2D(image, -1, self.sharpen_kernel)
        
        # Ensure pixel values are in valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def preprocess_image(self, image: np.ndarray, 
                        apply_blur: bool = True, 
                        apply_sharpening: bool = True) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to an image.
        
        Args:
            image: Input image as numpy array
            apply_blur: Whether to apply Gaussian blur
            apply_sharpening: Whether to apply sharpening
            
        Returns:
            Preprocessed image as numpy array
        """
        processed_image = image.copy()
        
        if apply_blur:
            processed_image = self.apply_gaussian_blur(processed_image)
        
        if apply_sharpening:
            processed_image = self.apply_sharpening(processed_image)
        
        return processed_image
    
    def process_image_file(self, input_path: Union[str, Path], 
                          output_path: Union[str, Path],
                          apply_blur: bool = True,
                          apply_sharpening: bool = True) -> bool:
        """
        Process a single image file and save the result.
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            apply_blur: Whether to apply Gaussian blur
            apply_sharpening: Whether to apply sharpening
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the image
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"Error: Could not read image {input_path}")
                return False
            
            # Apply preprocessing
            processed_image = self.preprocess_image(image, apply_blur, apply_sharpening)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed image
            success = cv2.imwrite(str(output_path), processed_image)
            if not success:
                print(f"Error: Could not save image to {output_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         file_pattern: str = "*.png",
                         apply_blur: bool = True,
                         apply_sharpening: bool = True) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            file_pattern: File pattern to match (e.g., "*.png", "frame_*_bg_subtracted.png")
            apply_blur: Whether to apply Gaussian blur
            apply_sharpening: Whether to apply sharpening
            
        Returns:
            Number of successfully processed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory {input_dir} does not exist")
            return 0
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        image_files = list(input_path.glob(file_pattern))
        
        if not image_files:
            print(f"No files found matching pattern {file_pattern} in {input_dir}")
            return 0
        
        print(f"Found {len(image_files)} images to process")
        
        successful_count = 0
        for image_file in image_files:
            # Create output filename
            output_filename = f"{image_file.stem}_preprocessed{image_file.suffix}"
            output_file_path = output_path / output_filename
            
            # Process the image
            if self.process_image_file(image_file, output_file_path, apply_blur, apply_sharpening):
                successful_count += 1
                print(f"Processed: {image_file.name} -> {output_filename}")
            else:
                print(f"Failed to process: {image_file.name}")
        
        print(f"Successfully processed {successful_count}/{len(image_files)} images")
        return successful_count


def create_preprocessing_script():
    """
    Create a standalone script for preprocessing background subtracted images.
    """
    script_content = '''#!/usr/bin/env python3
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
        print(f"\\nPreprocessing completed successfully!")
        print(f"Processed {successful_count} images")
        print(f"Output saved to: {args.output_dir}")
    else:
        print("\\nNo images were processed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    script_path = Path(__file__).parent / "preprocess_bg_subtracted.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


if __name__ == "__main__":
    # Create the preprocessing script when this module is run directly
    script_path = create_preprocessing_script()
    print(f"Created preprocessing script: {script_path}")
