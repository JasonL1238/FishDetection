"""
Abstract base class for all masking methods.
"""
from abc import ABC, abstractmethod


class BaseMasker(ABC):
    """
    Abstract base class for all masking methods.

    Subclasses must implement process_frame() and process_video().
    """

    @abstractmethod
    def process_frame(self, frame):
        """Process a single frame and return the masked result."""
        pass

    @abstractmethod
    def process_video(self, video_path, output_path):
        """Process an entire video file."""
        pass
