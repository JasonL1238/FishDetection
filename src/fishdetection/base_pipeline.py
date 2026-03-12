"""
Abstract base class for fish detection pipelines.

Defines the interface that all pipeline variants must implement:
subclasses provide a background model creation strategy.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple

from .background_subtractor import BackgroundSubtractor


class BasePipeline(ABC):
    """
    Base class for fish detection pipelines.

    Subclasses must implement create_background_model_strategy().
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
        self.fps = fps
        self.num_columns = num_columns
        self.target_per_column = target_per_column
        self.threshold = threshold
        self.morph_kernel_size = morph_kernel_size
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_object_size = min_object_size

        self.bg_subtractor = None
        self.hsv_masker = None
        self.video = None

    @abstractmethod
    def create_background_model_strategy(
        self,
        video_path: Path,
        start_frame: int,
        num_frames: int,
        output_dir: Path,
    ) -> BackgroundSubtractor:
        """
        Create and return a configured BackgroundSubtractor with a
        loaded background model.
        """
        pass
