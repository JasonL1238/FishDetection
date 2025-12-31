"""
Segment-Specific Background Pipeline - Solution 2

Uses frame indices within or near each segment for background model creation.
This adapts to background changes over time.
"""

from pathlib import Path
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from ..base.base_pipeline import BasePipeline


class SegmentBackgroundPipeline(BasePipeline):
    """
    Segment-specific background model pipeline.
    
    Uses frame indices within or near the segment being processed for background
    model creation. This adapts to background changes over time.
    """
    
    def create_background_model_strategy(self,
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create background model using frame indices within the segment.
        
        Uses frames from within the segment (with some padding) to create
        a background model that matches the current segment.
        """
        bg_subtractor = TrackingProgramBackgroundSubtractorV2(
            threshold=self.threshold,
            morph_kernel_size=self.morph_kernel_size
        )
        
        # Create frame indices within the segment
        # Use frames distributed throughout the segment
        end_frame = start_frame + num_frames
        segment_frames = list(range(start_frame, end_frame))
        
        # Sample 12 frames evenly distributed across the segment
        if len(segment_frames) >= 12:
            step = len(segment_frames) // 12
            frame_indices = [segment_frames[i * step] for i in range(12)]
        else:
            # If segment is too short, use all frames
            frame_indices = segment_frames
        
        print(f"Creating segment-specific background model from frames: {frame_indices[:3]}...{frame_indices[-3:]}")
        bg_subtractor.create_background_model(video_path, frame_indices=frame_indices)
        bg_subtractor.save_background_model(output_dir / "background_model.npy")
        
        return bg_subtractor



