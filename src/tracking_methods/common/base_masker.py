"""
Base masker class for all masking methods.

This module contains the abstract base class that all masking
methods should inherit from.
"""
from abc import ABC, abstractmethod


class BaseMasker(ABC):
    """
    Abstract base class for all masking methods.
    
    This class defines the common interface that all masking
    methods must implement.
    """
    
    @abstractmethod
    def process_frame(self, frame):
        """
        Process a single frame using the specific masking method.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with applied masking
        """
        pass
    
    @abstractmethod
    def process_video(self, video_path, output_path):
        """
        Process an entire video using the specific masking method.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
        """
        pass
