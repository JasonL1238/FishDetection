"""Fish detection module for identifying tiny black fish in images."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class FishDetection:
    """Represents a detected fish with bounding box and confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    area: int


class FishDetector:
    """Detector for tiny black fish with filtering to exclude bars and other objects."""
    
    def __init__(self, 
                 min_area: int = 50,
                 max_area: int = 2000,
                 min_aspect_ratio: float = 0.3,
                 max_aspect_ratio: float = 3.0,
                 black_threshold: int = 50):
        """
        Initialize the fish detector.
        
        Args:
            min_area: Minimum area for fish detection (pixels)
            max_area: Maximum area for fish detection (pixels)
            min_aspect_ratio: Minimum width/height ratio for fish shape
            max_aspect_ratio: Maximum width/height ratio for fish shape
            black_threshold: Threshold for black color detection (0-255)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.black_threshold = black_threshold
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for fish detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to better handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_black_objects(self, image: np.ndarray) -> List[FishDetection]:
        """Detect black objects that could be fish."""
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio (fish should be roughly oval/elongated)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Check if the object is actually black in the original image
            if self._is_black_object(image, x, y, w, h):
                # Calculate confidence based on area and shape
                confidence = self._calculate_confidence(contour, area, aspect_ratio)
                
                detection = FishDetection(
                    x=x, y=y, width=w, height=h, 
                    confidence=confidence, area=int(area)
                )
                detections.append(detection)
        
        return detections
    
    def _is_black_object(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Check if the object in the bounding box is actually black."""
        # Extract the region of interest
        roi = image[y:y+h, x:x+w]
        
        if len(roi) == 0:
            return False
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Calculate mean brightness
        mean_brightness = np.mean(roi_gray)
        
        # Object is considered black if mean brightness is below threshold
        return mean_brightness < self.black_threshold
    
    def _calculate_confidence(self, contour: np.ndarray, area: int, aspect_ratio: float) -> float:
        """Calculate confidence score for a detection."""
        # Calculate contour properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        # Circularity (4π*area/perimeter²) - fish should be somewhat circular/oval
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Ideal circularity for fish is around 0.7-0.9
        circularity_score = 1.0 - abs(circularity - 0.8) / 0.8
        
        # Aspect ratio score (prefer moderate elongation)
        aspect_score = 1.0 - abs(aspect_ratio - 1.5) / 1.5
        
        # Area score (prefer medium-sized objects)
        area_score = 1.0 - abs(area - (self.min_area + self.max_area) / 2) / ((self.max_area - self.min_area) / 2)
        
        # Combine scores
        confidence = (circularity_score * 0.4 + aspect_score * 0.3 + area_score * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def filter_detections(self, detections: List[FishDetection]) -> List[FishDetection]:
        """Filter detections to remove likely non-fish objects."""
        filtered = []
        
        for detection in detections:
            # Additional filtering based on shape characteristics
            if self._is_likely_fish(detection):
                filtered.append(detection)
        
        # Remove overlapping detections (non-maximum suppression)
        filtered = self._non_maximum_suppression(filtered)
        
        return filtered
    
    def _is_likely_fish(self, detection: FishDetection) -> bool:
        """Additional checks to determine if detection is likely a fish."""
        # Fish should have reasonable dimensions
        if detection.width < 5 or detection.height < 5:
            return False
        
        # Fish should not be too rectangular (bars are very rectangular)
        aspect_ratio = detection.width / detection.height
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
        
        return True
    
    def _non_maximum_suppression(self, detections: List[FishDetection], overlap_threshold: float = 0.3) -> List[FishDetection]:
        """Remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for detection in detections:
                if self._calculate_overlap(current, detection) < overlap_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def _calculate_overlap(self, det1: FishDetection, det2: FishDetection) -> float:
        """Calculate overlap ratio between two detections."""
        # Calculate intersection
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x + det1.width, det2.x + det2.width)
        y2 = min(det1.y + det1.height, det2.y + det2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = det1.area + det2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_fish(self, image_path: str) -> List[FishDetection]:
        """Main method to detect fish in an image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect black objects
        detections = self.detect_black_objects(image)
        
        # Filter detections
        filtered_detections = self.filter_detections(detections)
        
        return filtered_detections
    
    def visualize_detections(self, image_path: str, detections: List[FishDetection], 
                           output_path: Optional[str] = None) -> np.ndarray:
        """Visualize detections on the image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Draw detections
        for detection in detections:
            # Draw bounding box
            cv2.rectangle(image, 
                         (detection.x, detection.y), 
                         (detection.x + detection.width, detection.y + detection.height),
                         (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Fish: {detection.confidence:.2f}"
            cv2.putText(image, label, 
                       (detection.x, detection.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save or return image
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image



