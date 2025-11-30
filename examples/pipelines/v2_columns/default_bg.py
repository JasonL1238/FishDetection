"""
Default Background Pipeline - Current Implementation

Uses default frame indices [0, 2000, 4000, ...] for background model creation.
This is the original implementation that shows accuracy degradation over time.
"""

from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from ..base.base_pipeline import BasePipeline


class DefaultBackgroundPipeline(BasePipeline):
    """
    Default background model pipeline.
    
    Uses the default frame indices from the entire video for background model creation.
    This can cause accuracy issues for later segments due to background drift.
    """
    
    def create_background_model_strategy(self,
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create background model using default frame indices.
        
        Uses frames from the entire video, which may not match the segment being processed.
        """
        bg_subtractor = TrackingProgramBackgroundSubtractorV2(
            threshold=self.threshold,
            morph_kernel_size=self.morph_kernel_size
        )
        
        # Create background model using default frame indices
        print("Creating background model using default frame indices...")
        bg_subtractor.create_background_model(video_path)
        bg_subtractor.save_background_model(output_dir / "background_model.npy")
        
        return bg_subtractor

