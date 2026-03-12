"""
Fish detection and tracking package.

Provides background subtraction, HSV masking, and grid-based
fish detection for video analysis.
"""

from .background_subtractor import BackgroundSubtractor
from .hsv_masker import HSVMasker
from .pipeline import CustomGridPipeline

__all__ = ['BackgroundSubtractor', 'HSVMasker', 'CustomGridPipeline']
