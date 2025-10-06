"""
Factory for creating masking method instances.

This module provides a factory pattern for creating instances
of different masking methods.
"""

from .optical_flow.optical_flow_masker import OpticalFlowMasker
from .canny_edge_detection.canny_masker import CannyMasker
from .yolo_tracking.yolo_tracker import YOLOTracker
from .background_subtraction.background_subtraction_masker import BackgroundSubtractionMasker
from .hsv_masking.hsv_masker import HSVMasker


class MaskerFactory:
    """
    Factory class for creating masking method instances.
    """
    
    @staticmethod
    def create_masker(method_name):
        """
        Create a masker instance based on the method name.
        
        Args:
            method_name: Name of the masking method ('optical_flow', 'canny', or 'yolo')
            
        Returns:
            Instance of the specified masker
            
        Raises:
            ValueError: If method_name is not supported
        """
        if method_name == 'optical_flow':
            return OpticalFlowMasker()
        elif method_name == 'canny':
            return CannyMasker()
        elif method_name == 'yolo':
            return YOLOTracker()
        elif method_name == 'background_subtraction':
            return BackgroundSubtractionMasker()
        elif method_name == 'hsv':
            return HSVMasker()
        else:
            raise ValueError(f"Unsupported masking method: {method_name}")
    
    @staticmethod
    def get_available_methods():
        """
        Get list of available masking methods.
        
        Returns:
            List of available masking method names
        """
        return ['optical_flow', 'canny', 'yolo', 'background_subtraction', 'hsv']
