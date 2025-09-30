"""
Processing modules for fish detection pipeline.

This package contains reusable processing components that can be used
across different tracking methods and analysis pipelines.
"""

from .background_subtractor import BackgroundSubtractor

__all__ = ['BackgroundSubtractor']
