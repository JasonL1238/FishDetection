"""
Optical flow-based masker implementation.

This module contains the main OpticalFlowMasker class for detecting
fish using optical flow analysis based on the tutorial from Isaac Berrios.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from ..common.base_masker import BaseMasker


class OpticalFlowMasker(BaseMasker):
    """
    Masker that uses optical flow to detect fish movement.
    
    This class implements optical flow-based masking techniques
    for fish detection in video sequences using Farneback's method.
    """
    
    def __init__(self, 
                 pyr_scale: float = 0.75,
                 levels: int = 3,
                 winsize: int = 5,
                 iterations: int = 3,
                 poly_n: int = 10,
                 poly_sigma: float = 1.2,
                 motion_thresh: float = 1.0,
                 bbox_thresh: int = 400,
                 nms_thresh: float = 0.1,
                 kernel_size: Tuple[int, int] = (7, 7),
                 use_variable_threshold: bool = True,
                 min_thresh: float = 0.3,
                 max_thresh: float = 1.0):
        """
        Initialize the optical flow masker.
        
        Args:
            pyr_scale: Parameter specifying the image scale (<1) to build pyramids
            levels: Number of pyramid levels including the initial image
            winsize: Average window size
            iterations: Number of iterations the algorithm does at each pyramid level
            poly_n: Size of the pixel neighborhood used to find polynomial expansion
            poly_sigma: Standard deviation of the Gaussian that is used to smooth derivatives
            motion_thresh: Minimum flow threshold for motion detection
            bbox_thresh: Minimum threshold area for declaring a bounding box
            nms_thresh: IOU threshold for computing Non-Maximal Suppression
            kernel_size: Kernel size for morphological operations
            use_variable_threshold: Whether to use variable threshold based on image position
            min_thresh: Minimum threshold for variable thresholding
            max_thresh: Maximum threshold for variable thresholding
        """
        super().__init__()
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.motion_thresh = motion_thresh
        self.bbox_thresh = bbox_thresh
        self.nms_thresh = nms_thresh
        self.kernel_size = kernel_size
        self.use_variable_threshold = use_variable_threshold
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        
        # Create morphological kernel
        self.kernel = np.ones(kernel_size, dtype=np.uint8)
        
        # Store previous frame for optical flow computation
        self.prev_frame = None
        self.prev_gray = None
        
        # Variable threshold for different image regions
        self.motion_thresh_array = None
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames using Farneback's method.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Optical flow as numpy array
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2.copy()
        
        # Apply Gaussian blur
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 5)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 5)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        
        return flow
    
    def get_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        """
        Obtain BGR image to visualize the optical flow.
        
        Args:
            flow: Optical flow array
            
        Returns:
            RGB visualization of optical flow
        """
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    
    def get_motion_mask(self, flow_mag: np.ndarray, 
                       motion_thresh: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Obtain motion mask from optical flow magnitude.
        
        Args:
            flow_mag: Optical flow magnitude
            motion_thresh: Threshold array (if None, uses self.motion_thresh)
            
        Returns:
            Binary motion mask
        """
        if motion_thresh is None:
            if self.use_variable_threshold and self.motion_thresh_array is not None:
                motion_thresh = self.motion_thresh_array
            else:
                motion_thresh = self.motion_thresh
        
        # Create motion mask
        motion_mask = np.uint8(flow_mag > motion_thresh) * 255
        
        # Apply morphological operations
        motion_mask = cv2.erode(motion_mask, self.kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        
        return motion_mask
    
    def get_contour_detections(self, mask: np.ndarray, 
                              flow_angle: Optional[np.ndarray] = None,
                              angle_thresh: float = 2.0) -> List[Tuple[int, int, int, int, float]]:
        """
        Obtain detections from contours in the motion mask.
        
        Args:
            mask: Binary motion mask
            flow_angle: Optical flow angle array (optional)
            angle_thresh: Threshold for flow angle standard deviation
            
        Returns:
            List of detections as (x1, y1, x2, y2, area) tuples
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        detections = []
        temp_mask = np.zeros_like(mask)
        
        if flow_angle is not None:
            angle_thresh = angle_thresh * flow_angle.std()
        
        for cnt in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Optional: Filter by flow angle consistency
            if flow_angle is not None:
                cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
                flow_angle_region = flow_angle[temp_mask > 0]
                temp_mask.fill(0)  # Reset temp mask
                
                if len(flow_angle_region) > 0 and flow_angle_region.std() > angle_thresh:
                    continue
            
            # Filter by area
            if area > self.bbox_thresh:
                detections.append((x, y, x + w, y + h, area))
        
        return detections
    
    def non_max_suppression(self, boxes: List[Tuple[int, int, int, int, float]], 
                           threshold: float) -> List[Tuple[int, int, int, int, float]]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            boxes: List of detections as (x1, y1, x2, y2, area) tuples
            threshold: IOU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(boxes) == 0:
            return []
        
        # Convert to numpy array for easier processing
        boxes = np.array(boxes)
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores (descending)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the box with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            remaining = indices[1:]
            
            # Calculate intersection coordinates
            xx1 = np.maximum(x1[current], x1[remaining])
            yy1 = np.maximum(y1[current], y1[remaining])
            xx2 = np.minimum(x2[current], x2[remaining])
            yy2 = np.minimum(y2[current], y2[remaining])
            
            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            # Calculate IoU
            union = areas[current] + areas[remaining] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            indices = remaining[iou <= threshold]
        
        return boxes[keep].tolist()
    
    def setup_variable_threshold(self, height: int, width: int):
        """
        Setup variable threshold array based on image dimensions.
        
        Args:
            height: Image height
            width: Image width
        """
        if self.use_variable_threshold:
            # Create linear threshold from min_thresh at top to max_thresh at bottom
            self.motion_thresh_array = np.c_[np.linspace(self.min_thresh, self.max_thresh, height)].repeat(width, axis=-1)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        Process a single frame using optical flow.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Tuple of (motion_mask, detections) where detections are (x1, y1, x2, y2, area) tuples
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Setup variable threshold if first frame
        if self.prev_gray is None:
            self.setup_variable_threshold(gray.shape[0], gray.shape[1])
        
        # Skip if no previous frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray), []
        
        # Compute optical flow
        flow = self.compute_flow(self.prev_gray, gray)
        
        # Separate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get motion mask
        motion_mask = self.get_motion_mask(mag)
        
        # Get detections from contours
        detections = self.get_contour_detections(motion_mask, ang)
        
        # Apply non-maximum suppression
        if detections:
            detections = self.non_max_suppression(detections, self.nms_thresh)
        
        # Update previous frame
        self.prev_gray = gray
        
        return motion_mask, detections
    
    def get_detections(self, frame1: np.ndarray, frame2: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Get detections from two consecutive frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            List of detections as (x1, y1, x2, y2, area) tuples
        """
        # Compute optical flow
        flow = self.compute_flow(frame1, frame2)
        
        # Separate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get motion mask
        motion_mask = self.get_motion_mask(mag)
        
        # Get detections from contours
        detections = self.get_contour_detections(motion_mask, ang)
        
        # Apply non-maximum suppression
        if detections:
            detections = self.non_max_suppression(detections, self.nms_thresh)
        
        return detections
    
    def process_video(self, video_path: str, output_path: str):
        """
        Process an entire video using optical flow detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
        """
        import cv2
        import pims
        
        # Load video
        video = pims.PyAVReaderIndexed(video_path)
        
        # Get video properties
        height, width = video[0].shape[:2]
        fps = 20  # Default FPS
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        for i, frame in enumerate(video):
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Process with optical flow
            motion_mask, detections = self.process_frame(gray)
            
            # Create output frame with detections
            output_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2, area = detection
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, f"Area: {area:.0f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Write frame
            out.write(output_frame)
        
        # Release video writer
        out.release()
    
    def reset(self):
        """Reset the masker state."""
        self.prev_frame = None
        self.prev_gray = None
        self.motion_thresh_array = None
