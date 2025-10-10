"""
Tracking Program Background Subtraction Module

This module provides a specialized background subtraction implementation
extracted from the tracking program for fish detection applications.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union
from pathlib import Path


class TrackingProgramBackgroundSubtractor:
    """
    A specialized background subtraction class extracted from the tracking program.
    
    This class provides methods to create background models from video frames
    and apply background subtraction specifically optimized for fish detection.
    """
    
    def __init__(self, 
                 threshold: int = 25,
                 blur_kernel_size: tuple = (3, 3),
                 blur_sigma: float = 0,
                 default_frame_indices: Optional[List[int]] = None):
        """
        Initialize the tracking program background subtractor.
        
        Args:
            threshold: Threshold value for binary mask creation
            blur_kernel_size: Kernel size for Gaussian blur
            blur_sigma: Standard deviation for Gaussian blur
            default_frame_indices: Default frame indices for background model creation
        """
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        
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
        """
        Create a background model using specified frame indices.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to use for background model.
                          If None, uses default frame indices.
            
        Returns:
            Background model as numpy array (grayscale, float64)
        """
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
        Apply background subtraction to a single frame.
        
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
        
        return thresh.astype(np.uint8)
    
    def process_video_frames(self, 
                           video_path: Union[str, Path], 
                           num_frames: Optional[int] = None,
                           start_frame: int = 0) -> List[np.ndarray]:
        """
        Process multiple frames from a video with background subtraction.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to process. If None, processes all frames.
            start_frame: Starting frame index
            
        Returns:
            List of background subtracted frames
        """
        if self.background_model is None:
            raise ValueError("No background model available. Create one first using create_background_model()")
        
        print("Loading video for frame processing...")
        video = pims.PyAVReaderIndexed(str(video_path))
        
        if num_frames is None:
            num_frames = len(video) - start_frame
        
        # Limit to available frames
        num_frames = min(num_frames, len(video) - start_frame)
        
        print(f"Processing {num_frames} frames starting from frame {start_frame}...")
        
        bg_subtracted_frames = []
        for i in range(start_frame, start_frame + num_frames):
            if i < len(video):
                # Get frame and convert to grayscale
                frame = video[i][:, :, 0]
                
                # Apply background subtraction
                bg_subtracted = self.apply_background_subtraction(frame)
                bg_subtracted_frames.append(bg_subtracted)
        
        return bg_subtracted_frames
    
    def save_background_model(self, filepath: Union[str, Path]):
        """
        Save the background model to a file.
        
        Args:
            filepath: Path where to save the background model
        """
        if self.background_model is None:
            raise ValueError("No background model to save. Create one first using create_background_model()")
        
        np.save(str(filepath), self.background_model)
        print(f"Background model saved to: {filepath}")
    
    def load_background_model(self, filepath: Union[str, Path]):
        """
        Load a background model from a file.
        
        Args:
            filepath: Path to the saved background model file
        """
        self.background_model = np.load(str(filepath))
        print(f"Background model loaded from: {filepath}")
    
    def get_background_model(self) -> Optional[np.ndarray]:
        """
        Get the current background model.
        
        Returns:
            Background model array or None if not created
        """
        return self.background_model
    
    def reset(self):
        """Reset the background subtractor by clearing the background model."""
        self.background_model = None
        print("Background subtractor reset")
    
    def __repr__(self):
        return (f"TrackingProgramBackgroundSubtractor(threshold={self.threshold}, "
                f"blur_kernel_size={self.blur_kernel_size}, "
                f"blur_sigma={self.blur_sigma}, "
                f"has_background_model={self.background_model is not None})")
