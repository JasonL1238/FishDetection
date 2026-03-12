from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .positions import FishTrajectory


def plot_xy_over_time(
    traj: FishTrajectory,
    out_path: Path,
    units: str = "mm",
    title_prefix: str = "Fish",
) -> None:
    """
    Save a simple x(t) and y(t) plot for one fish.

    NaNs in x/y will create breaks in the plotted lines.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = traj.time_sec
    x = traj.x
    y = traj.y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x, label="x", linewidth=1.5)
    ax.plot(t, y, label="y", linewidth=1.5)

    ax.set_xlabel("time (sec)")
    ax.set_ylabel(f"position ({units})")
    ax.set_title(f"{title_prefix} {traj.fish_id}: x/y over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
