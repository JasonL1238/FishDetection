"""
Canny edge detection-based masker implementation.

This module contains the main CannyMasker class for detecting
fish using Canny edge detection.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple

from ..common.base_masker import BaseMasker
from ..common.utils import preprocess_frame, postprocess_mask, save_processed_frame


class CannyMasker(BaseMasker):
    """
    Masker that uses Canny edge detection to identify fish boundaries.
    
    This class implements Canny edge detection-based masking techniques
    for fish detection in video sequences. It processes background subtracted
    images to detect fish edges and create masks.
    """
    
    def __init__(self, 
                 low_threshold: int = 50,
                 high_threshold: int = 150,
                 aperture_size: int = 3,
                 l2_gradient: bool = False,
                 blur_kernel_size: int = 5,
                 morphology_kernel_size: int = 3):
        """
        Initialize the Canny edge detection masker.
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            aperture_size: Aperture size for Sobel operator
            l2_gradient: Whether to use L2 gradient
            blur_kernel_size: Size of Gaussian blur kernel for preprocessing
            morphology_kernel_size: Size of morphological operation kernel
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        
        # Create morphological kernel
        self.morphology_kernel = np.ones(
            (morphology_kernel_size, morphology_kernel_size), 
            np.uint8
        )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using Canny edge detection.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with Canny edge detection-based masking
        """
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Apply Gaussian blur to reduce noise
        if self.blur_kernel_size > 0:
            processed_frame = cv2.GaussianBlur(
                processed_frame, 
                (self.blur_kernel_size, self.blur_kernel_size), 
                0
            )
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            processed_frame,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient
        )
        
        # Apply morphological operations to clean up the edges
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morphology_kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, self.morphology_kernel)
        
        # Postprocess the mask
        edges = postprocess_mask(edges)
        
        return edges
    
    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Process an entire video using Canny edge detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Save processed frame
                save_processed_frame(processed_frame, output_path, frame_count)
                
                frame_count += 1
                
        finally:
            cap.release()
    
    def process_directory(self, input_dir: str, output_dir: str, 
                        file_pattern: str = "*_preprocessed.png") -> None:
        """
        Process all preprocessed images in a directory.
        
        Args:
            input_dir: Directory containing preprocessed images
            output_dir: Directory to save processed images
            file_pattern: Pattern to match input files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all matching files
        image_files = list(input_path.glob(file_pattern))
        image_files.sort()  # Sort to ensure consistent processing order
        
        if not image_files:
            print(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            # Load image
            frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            
            if frame is None:
                print(f"Warning: Could not load image {image_file}")
                continue
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Generate output filename
            output_filename = f"frame_{i:03d}_canny_edges.png"
            output_file_path = output_path / output_filename
            
            # Save processed frame
            cv2.imwrite(str(output_file_path), processed_frame)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        print(f"Processing complete! Results saved to {output_dir}")
    
    def detect_fish_contours(self, frame: np.ndarray, 
                           min_area: int = 500,
                           max_area: int = 10000) -> list:
        """
        Detect fish contours from Canny edge detection.
        
        Args:
            frame: Input frame
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            
        Returns:
            List of detected contours
        """
        # Get edges
        edges = self.process_frame(frame)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def create_fish_mask(self, frame: np.ndarray, 
                        min_area: int = 500,
                        max_area: int = 10000) -> np.ndarray:
        """
        Create a binary mask highlighting detected fish regions.
        
        Args:
            frame: Input frame
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            
        Returns:
            Binary mask with detected fish regions
        """
        # Get fish contours
        contours = self.detect_fish_contours(frame, min_area, max_area)
        
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Fill contours in mask
        cv2.fillPoly(mask, contours, 255)
        
        return mask
    
    def visualize_detection(self, frame: np.ndarray, 
                          min_area: int = 500,
                          max_area: int = 10000) -> np.ndarray:
        """
        Create a visualization of detected fish with red outlines.
        
        Args:
            frame: Input frame
            min_area: Minimum contour area to consider (increased to filter small spots)
            max_area: Maximum contour area to consider
            
        Returns:
            Frame with detection visualization
        """
        # Get fish contours
        contours = self.detect_fish_contours(frame, min_area, max_area)
        
        # Create visualization
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()
        
        # Draw contours in red
        for contour in contours:
            # Draw contour in red (BGR format: (0, 0, 255))
            cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 2)
            
            # Add area label
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(vis_frame, f"Area: {area:.0f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        return vis_frame
