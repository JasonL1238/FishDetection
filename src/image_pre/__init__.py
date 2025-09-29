"""
Image preprocessing module for fish detection.

This module provides preprocessing functions including Gaussian blur and sharpening
for background subtracted images to improve fish detection accuracy.
"""

from .preprocessor import ImagePreprocessor

__all__ = ['ImagePreprocessor']
