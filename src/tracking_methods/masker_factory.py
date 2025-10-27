"""
Factory for creating masking method instances.

This module provides a factory pattern for creating instances
of different masking methods.
"""

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
            method_name: Name of the masking method ('hsv')
            
        Returns:
            Instance of the specified masker
            
        Raises:
            ValueError: If method_name is not supported
        """
        if method_name == 'hsv':
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
        return ['hsv']
