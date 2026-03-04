#!/usr/bin/env python3
"""Debug script: extract all frames from an episode and name them by sharpness score."""

import sys
from pathlib import Path

import cv2

FRAMES_DIR = Path(__file__).resolve().parent / "frames"
DEBUG_DIR = Path(__file__).resolve().parent / "debug"


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_scores.py <episode_stem>")
        print("Example: python debug_scores.py s1_e01_a")
        sys.exit(1)

    episode = sys.argv[1]
    episode_dir = FRAMES_DIR / episode

    if not episode_dir.exists():
        print(f"ERROR: {episode_dir} not found. Run extract first.")
        sys.exit(1)

    out_dir = DEBUG_DIR / episode
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(f for f in episode_dir.glob("s*_[0-9][0-9].jpg")
                    if "selected" not in f.name and "scores" not in f.name)

    if not frames:
        print(f"No candidate frames found in {episode_dir}")
        sys.exit(1)

    print(f"Scoring {len(frames)} frames from {episode}...")

    for frame_path in frames:
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        shot_tag = frame_path.stem.rsplit("_", 1)[0]  # e.g. s01
        shot_dir = out_dir / shot_tag
        shot_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(shot_dir / f"{score:.1f}.jpg"), cv2.imread(str(frame_path)))

    print(f"Done. {len(frames)} frames written to {out_dir}")


if __name__ == "__main__":
    main()
