# utils.py
"""
Utility functions for VisionTrack-YOLO.
Includes: frame drawing, zone management, video writing, snapshot saving.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def draw_fps(frame, fps):
    """Draw FPS counter on frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def draw_counts(frame, counts, start_y=50):
    """Draw object counts on frame."""
    x = 10
    y = start_y
    for cls, val in counts.items():
        cv2.putText(frame, f"{cls}: {val}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += 30


def save_frame(frame, out_dir, prefix="frame"):
    """Save current frame as image file."""
    ensure_dir(out_dir)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = Path(out_dir) / filename
    cv2.imwrite(str(path), frame)
    return str(path)


def draw_zone(frame, polygon, color=(0, 0, 255), thickness=2):
    """Draw restricted zone polygon on frame."""
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.putText(frame, "Restricted Zone", (polygon[0][0], polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def point_in_poly(point, polygon):
    """
    Check if point is inside polygon using cv2.pointPolygonTest.

    Args:
        point: (x, y) tuple
        polygon: list of (x, y) tuples

    Returns:
        bool: True if point is inside polygon
    """
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def init_video_writer(out_path, fourcc_str='mp4v', fps=20, frame_size=(640, 480)):
    """
    Initialize video writer for saving output.

    Args:
        out_path: Output file path
        fourcc_str: Video codec (default: 'mp4v')
        fps: Frames per second
        frame_size: (width, height) tuple

    Returns:
        cv2.VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    return writer