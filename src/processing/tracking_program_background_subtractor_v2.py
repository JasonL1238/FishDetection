"""
Tracking Program Background Subtraction Module V2

This module provides an improved background subtraction implementation
with morphological closing to reconnect split fish blobs.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union
from pathlib import Path


class TrackingProgramBackgroundSubtractorV2:
    """
    Improved background subtraction with morphological closing.
    
    This version reduces blob splitting by adding morphological
    closing after thresholding to reconnect split fish blobs.
    """
    
    def __init__(self, 
                 threshold: int = 15,
                 blur_kernel_size: tuple = (3, 3),
                 blur_sigma: float = 0,
                 morph_kernel_size: tuple = (5, 5),
                 default_frame_indices: Optional[List[int]] = None):
        """
        Initialize the improved background subtractor.
        
        Args:
            threshold: Threshold value for binary mask creation (default: 15)
            blur_kernel_size: Kernel size for Gaussian blur
            blur_sigma: Standard deviation for Gaussian blur
            morph_kernel_size: Kernel size for morphological closing
            default_frame_indices: Default frame indices for background model creation
        """
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.morph_kernel_size = morph_kernel_size
        
        # Default background frame indices (from original tracking programs)
        if default_frame_indices is None:
            self.default_frame_indices = [0, 2000, 4000, 6000, 8000, 10000, 
                                       12000, 14000, 16000, 18000, 20000, 22000]
        else:
            self.default_frame_indices = default_frame_indices
        
        # Background model (created when needed)
        self.background_model = None
    
    def create_background_model(self, 
                              video_path: Union[str, Path], 
                              frame_indices: Optional[List[int]] = None) -> np.ndarray:
        """Create a background model using specified frame indices."""
        if frame_indices is None:
            frame_indices = self.default_frame_indices
        
        print("Loading video for background model creation...")
        video = pims.PyAVReaderIndexed(str(video_path))
        
        # Extract frames for background model
        frames = []
        for idx in frame_indices:
            if idx < len(video):
                frame = video[idx][:, :, 0]  # Take only the first channel (grayscale)
                frames.append(frame)
        
        if not frames:
            raise ValueError("No valid frames found for background model creation")
        
        # Create background model using median
        print(f"Creating background model from {len(frames)} frames...")
        bg = np.median(np.stack(frames, axis=2), axis=2)
        
        # Apply Gaussian blur to smooth the background
        bg_smoothed = cv2.GaussianBlur(bg.astype(np.float64), 
                                     self.blur_kernel_size, 
                                     self.blur_sigma)
        
        # Store the background model
        self.background_model = bg_smoothed
        
        return bg_smoothed
    
    def apply_background_subtraction(self, 
                                   frame: np.ndarray, 
                                   background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply background subtraction to a single frame with morphological closing.
        
        Args:
            frame: Input frame (grayscale)
            background: Background model. If None, uses stored background model.
            
        Returns:
            Background subtracted frame as binary mask (uint8)
        """
        if background is None:
            if self.background_model is None:
                raise ValueError("No background model available. Create one first using create_background_model()")
            background = self.background_model
        
        # Convert frame to float64 for processing
        frame_float = frame.astype(np.float64)
        
        # Apply Gaussian blur to the frame
        frame_blurred = cv2.GaussianBlur(frame_float, 
                                       self.blur_kernel_size, 
                                       self.blur_sigma)
        
        # Calculate absolute difference
        mask = np.abs(background - frame_blurred)
        
        # Apply threshold to create binary mask
        _, thresh = cv2.threshold(mask, self.threshold, 255, cv2.THRESH_BINARY)
        
        # IMPROVEMENT: Add morphological closing to reconnect split blobs
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh.astype(np.uint8)
    
    def save_background_model(self, filepath: Union[str, Path]):
        """Save the background model to a file."""
        if self.background_model is None:
            raise ValueError("No background model to save. Create one first using create_background_model()")
        
        np.save(str(filepath), self.background_model)
        print(f"Background model saved to: {filepath}")
    
    def load_background_model(self, filepath: Union[str, Path]):
        """Load a background model from a file."""
        self.background_model = np.load(str(filepath))
        print(f"Background model loaded from: {filepath}")
    
    def get_background_model(self) -> Optional[np.ndarray]:
        """Get the current background model."""
        return self.background_model
    
    def __repr__(self):
        return (f"TrackingProgramBackgroundSubtractorV2(threshold={self.threshold}, "
                f"morph_kernel_size={self.morph_kernel_size}, "
                f"has_background_model={self.background_model is not None})")

