"""
V2 Column-based pipeline implementations.

Each variant implements a different strategy for fish detection and column limiting.
"""

from .default_bg import DefaultBackgroundPipeline
from .shared_bg import SharedBackgroundPipeline
from .segment_bg import SegmentBackgroundPipeline
from .adaptive_column_limit import AdaptiveColumnLimitPipeline

__all__ = [
    'DefaultBackgroundPipeline',
    'SharedBackgroundPipeline', 
    'SegmentBackgroundPipeline',
    'AdaptiveColumnLimitPipeline'
]

