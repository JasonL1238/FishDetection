"""
Common utility functions for masking methods.

This module contains shared utility functions used across
different masking approaches.
"""

import cv2
import numpy as np


def preprocess_frame(frame):
    """
    Preprocess a frame for masking operations.
    
    Args:
        frame: Input frame
        
    Returns:
        Preprocessed frame
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame


def postprocess_mask(mask):
    """
    Postprocess a mask to clean up noise and artifacts.
    
    Args:
        mask: Input mask
        
    Returns:
        Cleaned mask
    """
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def save_processed_frame(frame, output_path, frame_number):
    """
    Save a processed frame to disk.
    
    Args:
        frame: Processed frame
        output_path: Directory to save the frame
        frame_number: Frame number for naming
    """
    filename = f"frame_{frame_number:03d}_processed.png"
    cv2.imwrite(f"{output_path}/{filename}", frame)
