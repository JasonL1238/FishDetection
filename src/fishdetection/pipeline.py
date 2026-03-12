"""
Custom grid pipeline for fish detection.

Divides each video frame into a 7x4 grid (28 cells), detects the
single largest blob per cell as the fish, and produces an annotated
output video plus a summary report.
"""

import csv
import cv2
import numpy as np
import pims
from datetime import datetime
from pathlib import Path
from typing import List

from .background_subtractor import BackgroundSubtractor
from .hsv_masker import HSVMasker
from .base_pipeline import BasePipeline
from .grid import get_grid_cells, find_largest_blob_in_cell


class CustomGridPipeline(BasePipeline):
    """
    Grid-based fish detection pipeline.

    For each frame:
      1. Background subtraction
      2. Divide frame into 28 grid cells
      3. Find the largest blob in each cell (= 1 fish per cell)
      4. Draw results and collect statistics
    """

    PIPELINE_NAME = "custom_grid"

    def __init__(self, *args, num_segments: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.target_per_cell = 1

    def create_background_model_strategy(
        self, video_path, start_frame, num_frames, output_dir
    ) -> BackgroundSubtractor:
        bg_subtractor = BackgroundSubtractor(
            threshold=self.threshold,
            morph_kernel_size=self.morph_kernel_size,
        )
        print("Creating background model using default frame indices...")
        bg_subtractor.create_background_model(video_path)
        bg_subtractor.save_background_model(output_dir / "background_model.npy")
        return bg_subtractor

    def process(self,
                video_path: Path,
                output_dir: Path,
                duration_seconds: int = 10,
                start_frame: int = 0) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        num_frames = int(self.fps * duration_seconds)

        print("Initializing background subtractor...")
        self.bg_subtractor = self.create_background_model_strategy(
            video_path, start_frame, num_frames, output_dir
        )

        print("Initializing HSV masker...")
        self.hsv_masker = HSVMasker(
            lower_hsv=self.hsv_lower,
            upper_hsv=self.hsv_upper,
            min_object_size=self.min_object_size,
            apply_morphology=True,
        )

        print(f"Loading video: {video_path}")
        self.video = pims.PyAVReaderIndexed(str(video_path))

        end_frame = min(start_frame + num_frames, len(self.video))
        actual_num_frames = end_frame - start_frame

        height, width = self.video[0].shape[:2]

        cells, row_boundaries = get_grid_cells(width, height)
        num_cells = len(cells)

        print(f"\nProcessing frames {start_frame} to {end_frame - 1} "
              f"({actual_num_frames} frames)")
        print(f"Grid: {num_cells} cells (7 columns x 4 rows)")
        print(f"Target: 1 fish per cell (largest blob)")
        print(f"Temporal segments: {self.num_segments}")

        self._run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = (f"{self._run_timestamp}_{self.PIPELINE_NAME}_"
                     f"{num_cells}cells_{self.num_segments}seg")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = output_dir / f"{base_name}.mp4"
        out = cv2.VideoWriter(
            str(video_out), fourcc, self.fps, (width, height), isColor=True
        )

        frames_per_segment = actual_num_frames // self.num_segments
        segment_boundaries = [
            start_frame + i * frames_per_segment
            for i in range(self.num_segments + 1)
        ]
        segment_boundaries[-1] = end_frame

        csv_sample_interval = int(self.fps * 2)
        csv_path = output_dir / f"{base_name}_positions.csv"
        csv_header = ["time_sec", "frame"]
        for fish_num in range(1, num_cells + 1):
            csv_header.extend([f"fish{fish_num}_x_norm", f"fish{fish_num}_y_norm"])
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        all_cell_counts: List[List[int]] = []
        total_fish_counts: List[int] = []
        frames_with_missing_fish: List[int] = []

        for seg_idx in range(self.num_segments):
            seg_start = segment_boundaries[seg_idx]
            seg_end = segment_boundaries[seg_idx + 1]
            seg_num_frames = seg_end - seg_start

            print(f"\n--- Segment {seg_idx + 1}/{self.num_segments} "
                  f"(frames {seg_start}..{seg_end - 1}, "
                  f"{seg_num_frames} frames) ---")

            for local_idx in range(seg_num_frames):
                global_idx = seg_start + local_idx
                frame = self.video[global_idx]

                if len(frame.shape) == 3:
                    gray_frame = frame[:, :, 0]
                else:
                    gray_frame = frame

                bg_subtracted = self.bg_subtractor.apply(gray_frame)

                all_contours = []
                all_centroids = []
                cell_numbers = []
                cell_counts = [0] * num_cells
                cell_centroids = [None] * num_cells

                for cell_idx, (x0, y0, x1, y1) in enumerate(cells):
                    contour, centroid = find_largest_blob_in_cell(
                        self.hsv_masker, bg_subtracted, x0, y0, x1, y1
                    )
                    if contour is not None and centroid is not None:
                        all_contours.append(contour)
                        all_centroids.append(centroid)
                        cell_numbers.append(cell_idx + 1)
                        cell_counts[cell_idx] = 1
                        cell_centroids[cell_idx] = centroid

                if (global_idx - start_frame) % csv_sample_interval == 0:
                    time_sec = (global_idx - start_frame) / self.fps
                    row = [f"{time_sec:.1f}", global_idx]
                    for ci in range(num_cells):
                        if cell_centroids[ci] is not None:
                            cy, cx = cell_centroids[ci]
                            x0, y0, x1, y1 = cells[ci]
                            cell_w = x1 - x0
                            cell_h = y1 - y0
                            x_norm = (cx - x0) / cell_w if cell_w > 0 else 0.0
                            y_norm = (y1 - cy) / cell_h if cell_h > 0 else 0.0
                            row.extend([round(x_norm, 4), round(y_norm, 4)])
                        else:
                            row.extend(["", ""])
                    csv_writer.writerow(row)

                all_cell_counts.append(cell_counts)
                total_fish_counts.append(len(all_centroids))

                if len(all_centroids) < num_cells:
                    frames_with_missing_fish.append(global_idx)

                output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

                for i in range(1, 7):
                    x = int(i * width / 7)
                    cv2.line(output_frame, (x, 0), (x, height),
                             (255, 255, 0), 2)

                for y in row_boundaries[1:-1]:
                    cv2.line(output_frame, (0, y), (width, y),
                             (255, 255, 0), 2)

                cv2.rectangle(output_frame, (0, 0), (width - 1, height - 1),
                              (255, 255, 0), 2)

                if local_idx == 0:
                    cv2.line(output_frame, (0, 0), (width, 0),
                             (0, 255, 255), 5)

                if all_contours:
                    cv2.drawContours(output_frame, all_contours, -1,
                                     (0, 255, 0), 2)

                for idx, (y, x) in enumerate(all_centroids):
                    cv2.circle(output_frame, (int(x), int(y)), 5,
                               (0, 0, 255), -1)
                    cv2.putText(output_frame, str(cell_numbers[idx]),
                                (int(x) + 8, int(y) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)

                total_fish = len(all_centroids)
                stats = [
                    (f"Segment: {seg_idx + 1}/{self.num_segments} | "
                     f"Frame: {local_idx:3d}/{seg_num_frames - 1} "
                     f"(Global: {global_idx})"),
                    f"Total Fish: {total_fish}/{num_cells} cells",
                    "Target: 1 fish per cell (largest blob)",
                ]
                for j, text in enumerate(stats):
                    cv2.putText(output_frame, text, (10, 30 + j * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                out.write(output_frame)

                if local_idx % 5 == 0:
                    fname = (f"{self._run_timestamp}_{self.PIPELINE_NAME}_"
                             f"frame_{global_idx:04d}_"
                             f"seg{seg_idx + 1}_cells.png")
                    cv2.imwrite(str(frames_dir / fname), output_frame)

                if local_idx % 20 == 0:
                    recent = (total_fish_counts[-20:]
                              if len(total_fish_counts) >= 20
                              else total_fish_counts)
                    print(f"  Segment {seg_idx + 1}, "
                          f"Frame {local_idx:3d}/{seg_num_frames - 1} "
                          f"(Global: {global_idx}): "
                          f"avg_total={np.mean(recent):.1f} fish")

        out.release()
        csv_file.close()
        print(f"Position CSV saved to: {csv_path}")

        self._generate_summary(
            output_dir, all_cell_counts, total_fish_counts,
            duration_seconds, start_frame, num_cells,
            frames_with_missing_fish,
        )

        print(f"\nResults saved to: {output_dir}")

    def _generate_summary(self, output_dir, all_cell_counts,
                          total_fish_counts, duration_seconds,
                          start_frame, num_cells,
                          frames_with_missing_fish):
        avg_total = np.mean(total_fish_counts)
        std_total = np.std(total_fish_counts)

        frames_with_all = sum(
            1 for counts in all_cell_counts if sum(counts) == num_cells
        )
        frames_missing = len(all_cell_counts) - frames_with_all
        n = len(total_fish_counts)

        base_name = (f"{self._run_timestamp}_{self.PIPELINE_NAME}_"
                     f"{num_cells}cells_{self.num_segments}seg")

        summary = f"""
{self.PIPELINE_NAME} Pipeline - Results
==================================
Run Date/Time: {self._run_timestamp}
Pipeline: {self.PIPELINE_NAME}

Configuration:
- Background Subtraction: threshold={self.threshold}, morph_kernel={self.morph_kernel_size}
- HSV Masking: lower={self.hsv_lower}, upper={self.hsv_upper}
- Grid: {num_cells} cells (7 columns x 4 rows)
- Detection: largest blob per cell (1 fish per cell)
- Temporal Segments: {self.num_segments}
- Frames Processed: {n} ({duration_seconds}s @ {self.fps} FPS)
- Start Frame: {start_frame}, End Frame: {start_frame + n - 1}

Fish Detection:
- Average Total Fish/Frame: {avg_total:.2f}
- Target Total: {num_cells}
- Std Dev: {std_total:.2f}
- Min: {np.min(total_fish_counts)}, Max: {np.max(total_fish_counts)}

Accuracy:
- All cells detected: {frames_with_all} ({frames_with_all / n * 100:.2f}%)
- At least one missing: {frames_missing} ({frames_missing / n * 100:.2f}%)

Output Files:
- Video: {base_name}.mp4
- Background model: background_model.npy
- Summary: this file
"""

        summary_path = output_dir / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)

        print(summary)
