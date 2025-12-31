"""
Wrapper functions for backward compatibility with existing code.

These functions maintain the same interface as the original implementation
while using the new modular pipeline structure.
"""

from pathlib import Path
from typing import Optional

from .default_bg import DefaultBackgroundPipeline
from .shared_bg import SharedBackgroundPipeline
from .segment_bg import SegmentBackgroundPipeline
from .adaptive_column_limit import AdaptiveColumnLimitPipeline
from .segmented_columns import SegmentedColumnsPipeline
from .half_sectioned import HalfSectionedPipeline


def process_complete_pipeline_v2_columns(video_path, output_dir, fps=20, num_columns=7, 
                                         target_per_column=4, duration_seconds=10, start_frame=0,
                                         variant: str = "default", num_segments: int = 7):
    """
    Process video with column-based detection (backward compatible wrapper).
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output
        fps: Frames per second
        num_columns: Number of columns to divide frame into
        target_per_column: Target number of fish per column
        duration_seconds: Duration in seconds to process
        start_frame: Starting frame number (default 0)
        variant: Pipeline variant to use ("default", "shared", "segment", "adaptive", "segmented", "half_sectioned")
        num_segments: Number of temporal segments (only used for "segmented" variant, default 7)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Select pipeline variant
    if variant == "default":
        pipeline = DefaultBackgroundPipeline(
            fps=fps,
            num_columns=num_columns,
            target_per_column=target_per_column
        )
    elif variant == "shared":
        pipeline = SharedBackgroundPipeline(
            fps=fps,
            num_columns=num_columns,
            target_per_column=target_per_column
        )
    elif variant == "segment":
        pipeline = SegmentBackgroundPipeline(
            fps=fps,
            num_columns=num_columns,
            target_per_column=target_per_column
        )
    elif variant == "adaptive":
        pipeline = AdaptiveColumnLimitPipeline(
            fps=fps,
            num_columns=num_columns,
            target_per_column=target_per_column
        )
    elif variant == "segmented":
        pipeline = SegmentedColumnsPipeline(
            fps=fps,
            num_columns=num_columns,
            target_per_column=target_per_column,
            num_segments=num_segments
        )
    elif variant == "half_sectioned":
        pipeline = HalfSectionedPipeline(
            fps=fps,
            num_columns=num_columns,  # Not used but required by BasePipeline
            target_per_column=target_per_column,  # Not used but required by BasePipeline
            num_segments=num_segments if num_segments is not None else 7,
            num_sections_per_half=7
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'default', 'shared', 'segment', 'adaptive', 'segmented', or 'half_sectioned'")
    
    # Process video
    pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        duration_seconds=duration_seconds,
        start_frame=start_frame
    )

