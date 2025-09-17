"""Enhanced preprocessing techniques for fish detection in challenging environments."""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import filters, morphology, feature
from skimage.filters import frangi, gabor
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label, regionprops


class EnhancedPreprocessor:
    """Advanced preprocessing techniques for tiny fish detection."""
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 frangi_sigma_range: Tuple[float, float] = (1, 3),
                 frangi_sigma_steps: int = 3,
                 gabor_frequency: float = 0.1,
                 gabor_theta: float = 0):
        """
        Initialize enhanced preprocessor.
        
        Args:
            clahe_clip_limit: CLAHE clipping limit
            clahe_tile_size: CLAHE tile grid size
            frangi_sigma_range: Frangi vesselness sigma range
            frangi_sigma_steps: Number of sigma steps for Frangi
            gabor_frequency: Gabor filter frequency
            gabor_theta: Gabor filter orientation angle
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.frangi_sigma_range = frangi_sigma_range
        self.frangi_sigma_steps = frangi_sigma_steps
        self.gabor_frequency = gabor_frequency
        self.gabor_theta = gabor_theta
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, 
            tileGridSize=clahe_tile_size
        )
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)
    
    def apply_tophat(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply top-hat morphological operation to emphasize dark thin objects."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply top-hat (white top-hat to emphasize dark objects)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        return tophat
    
    def apply_frangi_vesselness(self, image: np.ndarray) -> np.ndarray:
        """Apply Frangi vesselness filter to detect thin elongated structures."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        gray_norm = gray.astype(np.float64) / 255.0
        
        # Apply Frangi vesselness filter
        vesselness = frangi(
            gray_norm,
            scale_range=self.frangi_sigma_range,
            scale_step=self.frangi_sigma_steps,
            beta1=0.5,
            beta2=15,
            gamma=5,
            black_ridges=True  # Detect dark ridges (fish)
        )
        
        # Convert back to 0-255 range
        return (vesselness * 255).astype(np.uint8)
    
    def apply_gabor_filter(self, image: np.ndarray, 
                          frequency: Optional[float] = None,
                          theta: Optional[float] = None) -> np.ndarray:
        """Apply Gabor filter to detect oriented structures."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        gray_norm = gray.astype(np.float64) / 255.0
        
        # Apply Gabor filter
        freq = frequency if frequency is not None else self.gabor_frequency
        angle = theta if theta is not None else self.gabor_theta
        
        gabor_response = gabor(gray_norm, frequency=freq, theta=angle)[0]
        
        # Convert back to 0-255 range
        return (np.abs(gabor_response) * 255).astype(np.uint8)
    
    def apply_steerable_filters(self, image: np.ndarray) -> np.ndarray:
        """Apply steerable filters to detect oriented edges."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian derivatives in x and y directions
        gx = filters.gaussian(gray, sigma=1, order=(0, 1))
        gy = filters.gaussian(gray, sigma=1, order=(1, 0))
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        return (gradient_magnitude * 255).astype(np.uint8)
    
    def apply_adaptive_threshold(self, image: np.ndarray, 
                               method: str = 'gaussian',
                               block_size: int = 11,
                               c: int = 2) -> np.ndarray:
        """Apply adaptive thresholding with different methods."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'gaussian':
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, c
            )
        elif method == 'mean':
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY_INV, block_size, c
            )
        else:
            raise ValueError(f"Unknown adaptive threshold method: {method}")
    
    def apply_otsu_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Apply Otsu's thresholding method."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's method
        threshold_value, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        return binary, threshold_value
    
    def apply_morphological_operations(self, image: np.ndarray,
                                     open_kernel_size: int = 3,
                                     close_kernel_size: int = 5,
                                     iterations: int = 2) -> np.ndarray:
        """Apply morphological opening and closing operations."""
        # Create kernels
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        
        # Apply opening to remove noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
        
        # Apply closing to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
        
        return closed
    
    def apply_watershed_segmentation(self, image: np.ndarray, 
                                   min_distance: int = 10,
                                   min_area: int = 50) -> np.ndarray:
        """Apply watershed segmentation to separate touching objects."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima
        local_maxima = feature.peak_local_maxima(
            dist_transform, min_distance=min_distance, threshold_abs=0.3 * dist_transform.max()
        )
        
        # Create markers
        markers = np.zeros_like(dist_transform, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-dist_transform, markers, mask=binary)
        
        # Filter by area
        labeled = label(labels > 0)
        filtered = remove_small_objects(labeled, min_size=min_area)
        
        return (filtered > 0).astype(np.uint8) * 255
    
    def create_enhanced_mask(self, image: np.ndarray, 
                           method: str = 'combined',
                           **kwargs) -> np.ndarray:
        """Create enhanced binary mask using multiple techniques."""
        if method == 'clahe_adaptive':
            enhanced = self.apply_clahe(image)
            return self.apply_adaptive_threshold(enhanced, **kwargs)
        
        elif method == 'frangi_otsu':
            vesselness = self.apply_frangi_vesselness(image)
            binary, _ = self.apply_otsu_threshold(vesselness)
            return self.apply_morphological_operations(binary, **kwargs)
        
        elif method == 'tophat_adaptive':
            tophat = self.apply_tophat(image, **kwargs)
            return self.apply_adaptive_threshold(tophat, **kwargs)
        
        elif method == 'gabor_otsu':
            gabor_response = self.apply_gabor_filter(image, **kwargs)
            binary, _ = self.apply_otsu_threshold(gabor_response)
            return self.apply_morphological_operations(binary, **kwargs)
        
        elif method == 'combined':
            # Combine multiple methods
            clahe_enhanced = self.apply_clahe(image)
            tophat = self.apply_tophat(clahe_enhanced)
            frangi = self.apply_frangi_vesselness(clahe_enhanced)
            gabor = self.apply_gabor_filter(clahe_enhanced)
            
            # Combine responses
            combined = cv2.addWeighted(tophat, 0.4, frangi, 0.3, 0)
            combined = cv2.addWeighted(combined, 0.7, gabor, 0.3, 0)
            
            # Apply threshold
            binary, _ = self.apply_otsu_threshold(combined)
            return self.apply_morphological_operations(binary, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_fish_candidates(self, image: np.ndarray,
                             method: str = 'combined',
                             min_area: int = 20,
                             max_area: int = 800,
                             min_aspect_ratio: float = 2.0,
                             max_aspect_ratio: float = 8.0) -> list:
        """Detect fish candidates using enhanced preprocessing."""
        # Create enhanced mask
        mask = self.create_enhanced_mask(image, method=method)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / max(1, min(w, h))
            
            # Filter by aspect ratio
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # Calculate additional properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate confidence
            confidence = self._calculate_candidate_confidence(
                area, aspect_ratio, circularity, min_area, max_area
            )
            
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'confidence': confidence
            })
        
        return candidates
    
    def _calculate_candidate_confidence(self, area: int, aspect_ratio: float, 
                                      circularity: float, min_area: int, max_area: int) -> float:
        """Calculate confidence score for a fish candidate."""
        # Area score (prefer medium-sized objects)
        ideal_area = (min_area + max_area) / 2
        area_score = 1.0 - abs(area - ideal_area) / ideal_area
        
        # Aspect ratio score (prefer elongated shapes)
        aspect_score = min(1.0, aspect_ratio / 4.0)
        
        # Circularity score (prefer moderate circularity for fish)
        circularity_score = 1.0 - abs(circularity - 0.4) / 0.4
        
        # Combine scores
        confidence = (area_score * 0.3 + aspect_score * 0.4 + circularity_score * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def visualize_preprocessing_steps(self, image: np.ndarray, 
                                    output_path: Optional[str] = None) -> np.ndarray:
        """Visualize different preprocessing steps for debugging."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different preprocessing steps
        clahe_result = self.apply_clahe(image)
        tophat_result = self.apply_tophat(image)
        frangi_result = self.apply_frangi_vesselness(image)
        gabor_result = self.apply_gabor_filter(image)
        
        # Create combined visualization
        rows = 2
        cols = 3
        
        # Resize images to same size
        h, w = gray.shape[:2]
        target_size = (w // 2, h // 2)
        
        gray_resized = cv2.resize(gray, target_size)
        clahe_resized = cv2.resize(cv2.cvtColor(clahe_result, cv2.COLOR_BGR2GRAY) if len(clahe_result.shape) == 3 else clahe_result, target_size)
        tophat_resized = cv2.resize(tophat_result, target_size)
        frangi_resized = cv2.resize(frangi_result, target_size)
        gabor_resized = cv2.resize(gabor_result, target_size)
        
        # Create grid
        grid = np.zeros((h, w * 3), dtype=np.uint8)
        
        # Place images in grid
        grid[:h//2, :w//2] = gray_resized
        grid[:h//2, w//2:w] = clahe_resized
        grid[:h//2, w:w+w//2] = tophat_resized
        grid[h//2:, :w//2] = frangi_resized
        grid[h//2:, w//2:w] = gabor_resized
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, "Original", (10, 30), font, 0.7, 255, 2)
        cv2.putText(grid, "CLAHE", (w//2 + 10, 30), font, 0.7, 255, 2)
        cv2.putText(grid, "TopHat", (w + 10, 30), font, 0.7, 255, 2)
        cv2.putText(grid, "Frangi", (10, h//2 + 30), font, 0.7, 255, 2)
        cv2.putText(grid, "Gabor", (w//2 + 10, h//2 + 30), font, 0.7, 255, 2)
        
        if output_path:
            cv2.imwrite(output_path, grid)
        
        return grid
