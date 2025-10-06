"""
HSV Masker for Fish Tracking

This module provides HSV-based masking capabilities for fish detection
and tracking, particularly useful for background subtracted frames.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union, Tuple
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from ..common.base_masker import BaseMasker


class HSVMasker(BaseMasker):
    """
    HSV-based masker for fish detection and tracking.
    
    This class provides HSV color space masking capabilities that can be
    applied to background subtracted frames to better isolate fish objects
    based on their color characteristics.
    """
    
    def __init__(self, 
                 lower_hsv: Tuple[int, int, int] = (0, 30, 30),
                 upper_hsv: Tuple[int, int, int] = (180, 255, 255),
                 min_object_size: int = 25,
                 morph_kernel_size: int = 3,
                 apply_morphology: bool = True):
        """
        Initialize the HSV masker.
        
        Args:
            lower_hsv: Lower bound for HSV color range (H, S, V)
            upper_hsv: Upper bound for HSV color range (H, S, V)
            min_object_size: Minimum size for objects to be considered valid
            morph_kernel_size: Kernel size for morphological operations
            apply_morphology: Whether to apply morphological operations
        """
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.min_object_size = min_object_size
        self.morph_kernel_size = morph_kernel_size
        self.apply_morphology = apply_morphology
        
        # Create morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using HSV masking.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Processed frame with detected objects highlighted
        """
        # Convert to HSV if needed
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                # Assume BGR input
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            else:
                # Single channel input, convert to 3-channel first
                frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                hsv_frame = cv2.cvtColor(frame_3ch, cv2.COLOR_BGR2HSV)
        else:
            # Grayscale input, convert to 3-channel first
            frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            hsv_frame = cv2.cvtColor(frame_3ch, cv2.COLOR_BGR2HSV)
        
        # Create HSV mask
        mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations if enabled
        if self.apply_morphology:
            # Opening to remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            # Closing to fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Label connected components
        labeled = label(mask)
        
        # Remove small objects
        labeled_cleaned = remove_small_objects(labeled, self.min_object_size)
        
        # Re-label after cleaning
        labeled_final = label(labeled_cleaned)
        
        return labeled_final.astype(np.uint8)
    
    def process_background_subtracted_frame(self, bg_subtracted_frame: np.ndarray) -> np.ndarray:
        """
        Process a background subtracted frame using HSV masking.
        
        This method is specifically designed to work with background subtracted
        frames where fish appear as bright objects against a dark background.
        
        Args:
            bg_subtracted_frame: Background subtracted frame (grayscale)
            
        Returns:
            Processed frame with detected objects highlighted
        """
        # Convert grayscale to 3-channel BGR
        if len(bg_subtracted_frame.shape) == 2:
            frame_3ch = cv2.cvtColor(bg_subtracted_frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_3ch = bg_subtracted_frame
        
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame_3ch, cv2.COLOR_BGR2HSV)
        
        # For background subtracted frames, we want to detect bright objects
        # Adjust HSV range to detect bright areas
        lower_bright = np.array([0, 0, 100])  # Low hue, low sat, high value
        upper_bright = np.array([180, 255, 255])  # Any hue, any sat, high value
        
        # Create HSV mask for bright objects
        mask = cv2.inRange(hsv_frame, lower_bright, upper_bright)
        
        # Apply morphological operations
        if self.apply_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Label connected components
        labeled = label(mask)
        
        # Remove small objects
        labeled_cleaned = remove_small_objects(labeled, self.min_object_size)
        
        # Re-label after cleaning
        labeled_final = label(labeled_cleaned)
        
        return labeled_final.astype(np.uint8)
    
    def process_video(self, video_path: Union[str, Path], output_path: Union[str, Path]):
        """
        Process an entire video using HSV masking.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
        """
        print(f"Processing video with HSV masking: {video_path}")
        video = pims.PyAVReaderIndexed(str(video_path))
        
        # Get video properties
        height, width = video[0].shape[:2]
        fps = 20  # Default frame rate
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)
        
        try:
            for i, frame in enumerate(video):
                # Process frame
                processed = self.process_frame(frame)
                
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
        Detect fish centroids in a frame using HSV masking.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
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
    
    def detect_fish_centroids_bg_subtracted(self, bg_subtracted_frame: np.ndarray) -> List[tuple]:
        """
        Detect fish centroids in a background subtracted frame using HSV masking.
        
        Args:
            bg_subtracted_frame: Background subtracted frame (grayscale)
            
        Returns:
            List of (y, x) centroid coordinates
        """
        processed = self.process_background_subtracted_frame(bg_subtracted_frame)
        
        # Get region properties
        props = regionprops(processed)
        
        centroids = []
        for prop in props:
            if prop.area >= self.min_object_size:
                centroids.append(prop.centroid)
        
        return centroids
    
    def detect_fish_contours_and_centroids_bg_subtracted(self, bg_subtracted_frame: np.ndarray) -> Tuple[List[np.ndarray], List[tuple]]:
        """
        Detect fish contours and centroids in a background subtracted frame using HSV masking.
        
        Args:
            bg_subtracted_frame: Background subtracted frame (grayscale)
            
        Returns:
            Tuple of (contours, centroids) where contours is a list of contour arrays
            and centroids is a list of (y, x) centroid coordinates
        """
        processed = self.process_background_subtracted_frame(bg_subtracted_frame)
        
        # Use regionprops for more reliable detection
        from skimage.measure import regionprops
        
        # Get region properties
        props = regionprops(processed)
        
        valid_contours = []
        centroids = []
        
        # For each region, create a contour and get centroid
        for prop in props:
            if prop.area >= self.min_object_size:
                # Get the region's coordinates
                coords = prop.coords
                
                # Create a binary mask for this region
                region_mask = np.zeros_like(processed, dtype=np.uint8)
                for coord in coords:
                    region_mask[coord[0], coord[1]] = 255
                
                # Find contour for this region
                contours, _ = cv2.findContours(
                    region_mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Use the largest contour for this region
                    largest_contour = max(contours, key=cv2.contourArea)
                    valid_contours.append(largest_contour)
                    
                    # Use regionprops centroid (more reliable)
                    centroids.append(prop.centroid)
        
        return valid_contours, centroids
    
    def set_hsv_range(self, lower_hsv: Tuple[int, int, int], upper_hsv: Tuple[int, int, int]):
        """
        Update HSV color range for masking.
        
        Args:
            lower_hsv: Lower bound for HSV color range (H, S, V)
            upper_hsv: Upper bound for HSV color range (H, S, V)
        """
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
    
    def get_hsv_range(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get current HSV color range.
        
        Returns:
            Tuple of (lower_hsv, upper_hsv) ranges
        """
        return (tuple(self.lower_hsv), tuple(self.upper_hsv))
    
    def __repr__(self):
        return (f"HSVMasker(lower_hsv={tuple(self.lower_hsv)}, "
                f"upper_hsv={tuple(self.upper_hsv)}, "
                f"min_object_size={self.min_object_size}, "
                f"apply_morphology={self.apply_morphology})")
