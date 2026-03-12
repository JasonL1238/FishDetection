"""
Background subtraction with morphological closing.

Creates a median background model from sampled frames, then subtracts
it from each frame to isolate moving fish. Morphological closing
reconnects split blobs.
"""

import cv2
import numpy as np
import pims
from typing import List, Optional, Union
from pathlib import Path


class BackgroundSubtractor:
    """
    Background subtraction with morphological closing.

    Builds a median background from sampled frames, then per-frame:
    blur -> absolute difference -> threshold -> morphological closing.
    """

    def __init__(self,
                 threshold: int = 15,
                 blur_kernel_size: tuple = (3, 3),
                 blur_sigma: float = 0,
                 morph_kernel_size: tuple = (5, 5),
                 default_frame_indices: Optional[List[int]] = None):
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.morph_kernel_size = morph_kernel_size

        if default_frame_indices is None:
            self.default_frame_indices = [
                0, 2000, 4000, 6000, 8000, 10000,
                12000, 14000, 16000, 18000, 20000, 22000,
            ]
        else:
            self.default_frame_indices = default_frame_indices

        self.background_model = None

    def create_background_model(self,
                                video_path: Union[str, Path],
                                frame_indices: Optional[List[int]] = None) -> np.ndarray:
        """Create a median background model from sampled video frames."""
        if frame_indices is None:
            frame_indices = self.default_frame_indices

        print("Loading video for background model creation...")
        video = pims.PyAVReaderIndexed(str(video_path))

        frames = []
        for idx in frame_indices:
            if idx < len(video):
                frame = video[idx][:, :, 0]
                frames.append(frame)

        if not frames:
            raise ValueError("No valid frames found for background model creation")

        print(f"Creating background model from {len(frames)} frames...")
        bg = np.median(np.stack(frames, axis=2), axis=2)

        bg_smoothed = cv2.GaussianBlur(
            bg.astype(np.float64),
            self.blur_kernel_size,
            self.blur_sigma,
        )

        self.background_model = bg_smoothed
        return bg_smoothed

    def apply(self, frame: np.ndarray,
              background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Subtract background from a grayscale frame.

        Returns a binary mask (uint8): white = foreground, black = background.
        """
        if background is None:
            if self.background_model is None:
                raise ValueError(
                    "No background model available. "
                    "Call create_background_model() first."
                )
            background = self.background_model

        frame_float = frame.astype(np.float64)
        frame_blurred = cv2.GaussianBlur(
            frame_float, self.blur_kernel_size, self.blur_sigma
        )

        mask = np.abs(background - frame_blurred)
        _, thresh = cv2.threshold(mask, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.morph_kernel_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh.astype(np.uint8)

    # Keep old name as alias for compatibility
    apply_background_subtraction = apply

    def save_background_model(self, filepath: Union[str, Path]):
        """Save the background model to a .npy file."""
        if self.background_model is None:
            raise ValueError("No background model to save.")
        np.save(str(filepath), self.background_model)
        print(f"Background model saved to: {filepath}")

    def load_background_model(self, filepath: Union[str, Path]):
        """Load a background model from a .npy file."""
        self.background_model = np.load(str(filepath))
        print(f"Background model loaded from: {filepath}")

    def __repr__(self):
        return (
            f"BackgroundSubtractor(threshold={self.threshold}, "
            f"morph_kernel_size={self.morph_kernel_size}, "
            f"has_model={self.background_model is not None})"
        )
