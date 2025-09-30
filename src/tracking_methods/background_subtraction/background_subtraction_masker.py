"""
Background Subtraction Masker for Fish Tracking

This module provides a background subtraction-based masker that integrates
with the existing tracking framework.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from ..common.base_masker import BaseMasker
from ...processing.background_subtractor import BackgroundSubtractor


class BackgroundSubtractionMasker(BaseMasker):
    """
    Background subtraction masker for fish detection and tracking.
    
    This class integrates the BackgroundSubtractor with the tracking framework
    to provide fish detection capabilities using background subtraction.
    """
    
    def __init__(self, 
                 threshold: int = 25,
                 blur_kernel_size: tuple = (3, 3),
                 blur_sigma: float = 0,
                 min_object_size: int = 25,
                 frame_indices: Optional[List[int]] = None):
        """
        Initialize the background subtraction masker.
        
        Args:
            threshold: Threshold value for binary mask creation
            blur_kernel_size: Kernel size for Gaussian blur
            blur_sigma: Standard deviation for Gaussian blur
            min_object_size: Minimum size for objects to be considered valid
            frame_indices: Frame indices for background model creation
        """
        self.background_subtractor = BackgroundSubtractor(
            threshold=threshold,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            default_frame_indices=frame_indices
        )
        self.min_object_size = min_object_size
        self.background_model = None
    
    def create_background_model(self, video_path: Union[str, Path]):
        """
        Create background model from video.
        
        Args:
            video_path: Path to video file for background model creation
        """
        self.background_model = self.background_subtractor.create_background_model(video_path)
        print(f"Background model created with shape: {self.background_model.shape}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using background subtraction.
        
        Args:
            frame: Input frame (grayscale)
            
        Returns:
            Processed frame with detected objects highlighted
        """
        if self.background_model is None:
            raise ValueError("Background model not created. Call create_background_model() first.")
        
        # Apply background subtraction
        mask = self.background_subtractor.apply_background_subtraction(frame, self.background_model)
        
        # Label connected components
        labeled = label(mask)
        
        # Remove small objects
        labeled_cleaned = remove_small_objects(labeled, self.min_object_size)
        
        # Re-label after cleaning
        labeled_final = label(labeled_cleaned)
        
        return labeled_final.astype(np.uint8)
    
    def process_video(self, video_path: Union[str, Path], output_path: Union[str, Path]):
        """
        Process an entire video using background subtraction.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
        """
        if self.background_model is None:
            self.create_background_model(video_path)
        
        print(f"Processing video: {video_path}")
        video = pims.PyAVReaderIndexed(str(video_path))
        
        # Get video properties
        height, width = video[0].shape[:2]
        fps = 20  # Default frame rate
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)
        
        try:
            for i, frame in enumerate(video):
                # Convert to grayscale
                gray_frame = frame[:, :, 0] if len(frame.shape) == 3 else frame
                
                # Process frame
                processed = self.process_frame(gray_frame)
                
                # Convert to 3-channel for video writer
                processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                
                # Write frame
                out.write(processed_3ch)
                
                if i % 100 == 0:
                    print(f"Processed frame {i}/{len(video)}")
        
        finally:
            out.release()
        
        print(f"Processed video saved to: {output_path}")
    
    def detect_fish_centroids(self, frame: np.ndarray) -> List[tuple]:
        """
        Detect fish centroids in a frame using background subtraction.
        
        Args:
            frame: Input frame (grayscale)
            
        Returns:
            List of (y, x) centroid coordinates
        """
        processed = self.process_frame(frame)
        
        # Get region properties
        props = regionprops(processed)
        
        centroids = []
        for prop in props:
            if prop.area >= self.min_object_size:
                centroids.append(prop.centroid)
        
        return centroids
    
    def get_background_model(self) -> Optional[np.ndarray]:
        """Get the current background model."""
        return self.background_model
    
    def save_background_model(self, filepath: Union[str, Path]):
        """Save the background model to a file."""
        if self.background_model is not None:
            self.background_subtractor.background_model = self.background_model
            self.background_subtractor.save_background_model(filepath)
        else:
            raise ValueError("No background model to save")
    
    def load_background_model(self, filepath: Union[str, Path]):
        """Load a background model from a file."""
        self.background_subtractor.load_background_model(filepath)
        self.background_model = self.background_subtractor.get_background_model()
    
    def __repr__(self):
        return (f"BackgroundSubtractionMasker(threshold={self.background_subtractor.threshold}, "
                f"min_object_size={self.min_object_size}, "
                f"has_background_model={self.background_model is not None})")

