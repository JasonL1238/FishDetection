"""
Configuration settings for masking methods.

This module contains configuration parameters for different
masking approaches.
"""

# Optical Flow Configuration
OPTICAL_FLOW_CONFIG = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

# Canny Edge Detection Configuration
CANNY_CONFIG = {
    'low_threshold': 50,
    'high_threshold': 150,
    'aperture_size': 3,
    'l2gradient': False
}

# YOLOv8 Configuration
YOLO_CONFIG = {
    'model_path': 'yolov8n.pt',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.7,
    'max_track_length': 30,
    'track_buffer': 30
}

# Common Configuration
COMMON_CONFIG = {
    'output_format': 'png',
    'frame_prefix': 'frame',
    'frame_padding': 3
}
