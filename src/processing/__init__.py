"""
Processing modules for fish detection pipeline.

This package contains reusable processing components that can be used
across different tracking methods and analysis pipelines.
"""

from .tracking_program_background_subtractor import TrackingProgramBackgroundSubtractor

__all__ = ['TrackingProgramBackgroundSubtractor']
