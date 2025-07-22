"""
frame_extractor.py
------------------
Importable utility to extract a specified number of random, well-distributed frames
from a list of video files or folders containing video files.

Usage:
    from frame_extractor import extract_random_frames

    extract_random_frames(
        paths=["/videos", "/more/specific_video.mp4"],
        total_images=100,
        out_dir="frames_output",
        jpg_quality=90,
        seed=42
    )

Dependencies:
    pip install opencv-python numpy
"""

import os
import glob
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Video extensions allowed
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def _collect_video_files(paths: List[str]) -> List[str]:
    """
    Gather all video files from a list of files/folders.
    """
    collected: List[str] = []

    for entry in paths:
        p = Path(entry)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            collected.append(str(p))

        elif p.is_dir():
            for ext in VIDEO_EXTS:
                collected.extend(glob.glob(str(p / f"*{ext}")))
                collected.extend(glob.glob(str(p / f"*{ext.upper()}")))
        else:
            continue

    # Remove duplicates while preserving order
    seen = set()
    return [v for v in collected if not (v in seen or seen.add(v))]


def extract_random_frames(
    paths: List[str],
    total_images: int,
    out_dir: str = "extracted_frames",
    jpg_quality: int = 95,
    seed: int | None = None
) -> None:
    """
    Extract a total of `total_images` random frames from all video files found in `paths`.
    Videos can be individual files or folders containing video files.
    Images are saved as high-quality JPGs to `out_dir`.
    """
    if total_images <= 0:
        raise ValueError("total_images must be a positive integer")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    videos = _collect_video_files(paths)
    if not videos:
        raise FileNotFoundError("No video files found in the given paths.")

    os.makedirs(out_dir, exist_ok=True)

    frames_per_video, extra = divmod(total_images, len(videos))
    extracted = 0

    for idx, video_path in enumerate(videos):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Failed to open: {video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        target_frames = frames_per_video + (1 if idx < extra else 0)

        if frame_count == 0:
            print(f"[WARN] Skipping empty video: {video_path}")
            cap.release()
            continue

        indices = (
            list(range(frame_count))
            if target_frames >= frame_count
            else sorted(random.sample(range(frame_count), target_frames))
        )

        stem = Path(video_path).stem
        for i, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                print(f"[WARN] Could not read frame {frame_index} from {video_path}")
                continue

            timestamp = frame_index / fps
            filename = f"{stem}_frame_{frame_index:06}_t{timestamp:.2f}s_{extracted:06}.jpg"
            cv2.imwrite(
                os.path.join(out_dir, filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
            )
            extracted += 1

        cap.release()

    print(f"[✓] Extracted {extracted} frames to folder: {out_dir}")
