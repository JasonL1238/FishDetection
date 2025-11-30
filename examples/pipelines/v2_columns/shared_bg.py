"""
Shared Background Pipeline - Solution 1

Uses a single background model created from the beginning of the video
for all segments. This should maintain consistent accuracy across segments.
"""

from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from ..base.base_pipeline import BasePipeline


class SharedBackgroundPipeline(BasePipeline):
    """
    Shared background model pipeline.
    
    Uses a single background model created from the beginning of the video
    (frames 0-6000) for all segments. This maintains consistency and should
    prevent accuracy degradation.
    """
    
    def __init__(self, *args, shared_bg_path: Optional[Path] = None, **kwargs):
        """
        Initialize shared background pipeline.
        
        Args:
            shared_bg_path: Path to pre-created shared background model.
                          If None, will create one from frames 0-6000.
            *args, **kwargs: Passed to BasePipeline
        """
        super().__init__(*args, **kwargs)
        self.shared_bg_path = shared_bg_path
    
    def create_background_model_strategy(self,
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create or load shared background model from beginning of video.
        
        Uses frames 0-6000 (first 5 minutes) to create a consistent background
        model that works for all segments.
        """
        bg_subtractor = TrackingProgramBackgroundSubtractorV2(
            threshold=self.threshold,
            morph_kernel_size=self.morph_kernel_size
        )
        
        # Use shared background model
        if self.shared_bg_path and self.shared_bg_path.exists():
            print(f"Loading shared background model from: {self.shared_bg_path}")
            bg_subtractor.load_background_model(self.shared_bg_path)
        else:
            # Create shared background from first 5 minutes (frames 0-6000)
            print("Creating shared background model from frames 0-6000 (first 5 minutes)...")
            frame_indices = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
            bg_subtractor.create_background_model(video_path, frame_indices=frame_indices)
            
            # Save shared background if path provided
            if self.shared_bg_path:
                bg_subtractor.save_background_model(self.shared_bg_path)
        
        # Also save to output directory for reference
        bg_subtractor.save_background_model(output_dir / "background_model.npy")
        
        return bg_subtractor

