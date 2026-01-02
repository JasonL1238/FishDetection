"""
Base pipeline class for column-based fish detection.

This class provides the common processing logic that all pipeline variants share.
Each variant only needs to implement the background model creation strategy.
"""

import cv2
import numpy as np
import pims
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


class BasePipeline(ABC):
    """
    Base class for column-based fish detection pipelines.
    
    Subclasses should implement:
    - create_background_model_strategy() - How to create the background model
    """
    
    def __init__(self, 
                 fps: int = 20,
                 num_columns: int = 7,
                 target_per_column: int = 4,
                 threshold: int = 15,
                 morph_kernel_size: Tuple[int, int] = (5, 5),
                 hsv_lower: Tuple[int, int, int] = (0, 0, 100),
                 hsv_upper: Tuple[int, int, int] = (180, 255, 255),
                 min_object_size: int = 10):
        """Initialize the base pipeline."""
        self.fps = fps
        self.num_columns = num_columns
        self.target_per_column = target_per_column
        self.threshold = threshold
        self.morph_kernel_size = morph_kernel_size
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_object_size = min_object_size
        
        # Will be initialized in process()
        self.bg_subtractor = None
        self.hsv_masker = None
        self.video = None
    
    @abstractmethod
    def create_background_model_strategy(self, 
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create background model using a specific strategy.
        
        Args:
            video_path: Path to input video
            start_frame: Starting frame index
            num_frames: Number of frames to process
            output_dir: Output directory for saving background model
            
        Returns:
            TrackingProgramBackgroundSubtractorV2 instance
        """
        pass

