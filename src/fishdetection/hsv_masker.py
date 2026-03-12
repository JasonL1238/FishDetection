"""
HSV-based masking for fish detection.

Filters bright objects via HSV color space, applies morphological
cleanup, labels connected components, and extracts contours/centroids.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union, Tuple
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from .base_masker import BaseMasker


class HSVMasker(BaseMasker):
    """
    HSV-based masker for fish detection and tracking.

    Works on raw frames or on background-subtracted frames where fish
    appear as bright objects against a dark background.
    """

    def __init__(self,
                 lower_hsv: Tuple[int, int, int] = (0, 30, 30),
                 upper_hsv: Tuple[int, int, int] = (180, 255, 255),
                 min_object_size: int = 25,
                 morph_kernel_size: int = 3,
                 apply_morphology: bool = True):
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.min_object_size = min_object_size
        self.morph_kernel_size = morph_kernel_size
        self.apply_morphology = apply_morphology

        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )

    # ------------------------------------------------------------------
    # BaseMasker interface
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame using HSV masking. Returns labeled image."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            if len(frame.shape) == 2:
                frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            hsv_frame = cv2.cvtColor(frame_3ch, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)

        if self.apply_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        labeled = label(mask)
        labeled_cleaned = remove_small_objects(labeled, self.min_object_size)
        labeled_final = label(labeled_cleaned)

        return labeled_final.astype(np.uint8)

    def process_video(self, video_path: Union[str, Path],
                      output_path: Union[str, Path]):
        """Process an entire video using HSV masking."""
        print(f"Processing video with HSV masking: {video_path}")
        video = pims.PyAVReaderIndexed(str(video_path))

        height, width = video[0].shape[:2]
        fps = 20

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps,
                              (width, height), isColor=False)

        try:
            for i, frame in enumerate(video):
                processed = self.process_frame(frame)
                processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                out.write(processed_3ch)
                if i % 100 == 0:
                    print(f"Processed frame {i}/{len(video)}")
        finally:
            out.release()

        print(f"Processed video saved to: {output_path}")

    # ------------------------------------------------------------------
    # Background-subtracted frame processing
    # ------------------------------------------------------------------

    def process_background_subtracted_frame(self,
                                            bg_subtracted_frame: np.ndarray
                                            ) -> np.ndarray:
        """
        Process a background-subtracted frame. Detects bright objects
        (fish) against a dark background.
        """
        if len(bg_subtracted_frame.shape) == 2:
            frame_3ch = cv2.cvtColor(bg_subtracted_frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_3ch = bg_subtracted_frame

        hsv_frame = cv2.cvtColor(frame_3ch, cv2.COLOR_BGR2HSV)

        lower_bright = np.array([0, 0, 100])
        upper_bright = np.array([180, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_bright, upper_bright)

        if self.apply_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        labeled = label(mask)
        labeled_cleaned = remove_small_objects(labeled, self.min_object_size)
        labeled_final = label(labeled_cleaned)

        return labeled_final.astype(np.uint8)

    # ------------------------------------------------------------------
    # Centroid / contour detection
    # ------------------------------------------------------------------

    def detect_fish_centroids(self, frame: np.ndarray) -> List[tuple]:
        """Detect fish centroids in a raw frame. Returns list of (y, x)."""
        processed = self.process_frame(frame)
        props = regionprops(processed)
        return [p.centroid for p in props if p.area >= self.min_object_size]

    def detect_fish_centroids_bg_subtracted(self,
                                            bg_subtracted_frame: np.ndarray
                                            ) -> List[tuple]:
        """Detect fish centroids in a background-subtracted frame."""
        processed = self.process_background_subtracted_frame(bg_subtracted_frame)
        props = regionprops(processed)
        return [p.centroid for p in props if p.area >= self.min_object_size]

    def detect_fish_contours_and_centroids_bg_subtracted(
        self, bg_subtracted_frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[tuple]]:
        """
        Detect contours and centroids in a background-subtracted frame.

        Returns:
            (contours, centroids) — paired lists where contours[i]
            corresponds to centroids[i] as (y, x).
        """
        processed = self.process_background_subtracted_frame(bg_subtracted_frame)
        props = regionprops(processed)

        valid_contours = []
        centroids = []

        for prop in props:
            if prop.area < self.min_object_size:
                continue

            region_mask = np.zeros_like(processed, dtype=np.uint8)
            for coord in prop.coords:
                region_mask[coord[0], coord[1]] = 255

            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                valid_contours.append(largest_contour)
                centroids.append(prop.centroid)

        return valid_contours, centroids

    # ------------------------------------------------------------------
    # HSV range configuration
    # ------------------------------------------------------------------

    def set_hsv_range(self, lower_hsv: Tuple[int, int, int],
                      upper_hsv: Tuple[int, int, int]):
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)

    def get_hsv_range(self) -> Tuple[Tuple[int, int, int],
                                     Tuple[int, int, int]]:
        return (tuple(self.lower_hsv), tuple(self.upper_hsv))

    def __repr__(self):
        return (
            f"HSVMasker(lower_hsv={tuple(self.lower_hsv)}, "
            f"upper_hsv={tuple(self.upper_hsv)}, "
            f"min_object_size={self.min_object_size}, "
            f"apply_morphology={self.apply_morphology})"
        )
