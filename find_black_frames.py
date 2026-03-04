"""
Find black/near-black frames in Spy vs Spy v2 raw frame sequences.
These mark episode boundaries within the compilation.

Strategy: Two-pass approach
  Pass 1: Sample every 10th frame to find approximate boundary regions
  Pass 2: Fine scan around each detected region (+/- 50 frames)

If Pass 1 completes fast enough, we fall back to a full scan instead.
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

FRAMES_DIR = r"c:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy-v2\raw\frames"
THRESHOLD = 10  # mean pixel value below this = "black" frame

def get_frame_path(season: int, frame_num: int) -> str:
    """Build the filename for a given season and frame number."""
    # Files are named like s1_0001.png (4-digit zero-padded) up to s1_25462.png (no extra padding past 4)
    # Actually: 4-digit zero-padded for 1-9999, then just the number for 10000+
    # The pattern is: zero-padded to at least 4 digits
    name = f"s{season}_{frame_num:04d}.png" if frame_num < 10000 else f"s{season}_{frame_num}.png"
    return os.path.join(FRAMES_DIR, name)


def scan_frames(season: int, max_frame: int, step: int = 1, frame_range=None):
    """
    Scan frames and return list of (frame_num, mean_pixel_value) for black frames.

    Args:
        season: 1 or 2
        max_frame: last frame number
        step: step size between frames (1 for full scan, 10 for coarse)
        frame_range: optional list/range of specific frame numbers to check
    """
    black_frames = []

    if frame_range is not None:
        frames_to_check = frame_range
    else:
        frames_to_check = range(1, max_frame + 1, step)

    checked = 0
    for fnum in frames_to_check:
        if fnum < 1 or fnum > max_frame:
            continue
        path = get_frame_path(season, fnum)
        if not os.path.exists(path):
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        mean_val = np.mean(img)
        if mean_val < THRESHOLD:
            black_frames.append((fnum, mean_val))

        checked += 1
        if checked % 5000 == 0:
            print(f"  ... checked {checked} frames so far ({fnum})", flush=True)

    return black_frames


def print_results(season: int, black_frames: list):
    """Print black frames grouped by season with gaps."""
    print(f"\n{'='*70}")
    print(f"  SEASON {season} - Black/Near-Black Frames (mean < {THRESHOLD})")
    print(f"{'='*70}")

    if not black_frames:
        print("  No black frames found!")
        return

    # Sort by frame number
    black_frames.sort(key=lambda x: x[0])

    # Group consecutive black frames into clusters (boundaries)
    clusters = []
    current_cluster = [black_frames[0]]

    for i in range(1, len(black_frames)):
        fnum, val = black_frames[i]
        prev_fnum = black_frames[i-1][0]
        if fnum - prev_fnum <= 5:  # frames within 5 of each other = same cluster
            current_cluster.append(black_frames[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [black_frames[i]]
    clusters.append(current_cluster)

    print(f"\n  Found {len(black_frames)} black frames in {len(clusters)} clusters (episode boundaries)\n")

    # Print individual frames
    print(f"  {'Frame':>8}  {'Mean Px':>8}  {'Gap from prev':>14}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}")

    prev_frame = None
    for fnum, val in black_frames:
        gap_str = ""
        if prev_frame is not None:
            gap = fnum - prev_frame
            gap_str = str(gap)
        prev_frame = fnum
        print(f"  {fnum:>8}  {val:>8.2f}  {gap_str:>14}")

    # Print cluster summary
    print(f"\n  --- Cluster Summary (Episode Boundaries) ---")
    print(f"  {'#':>3}  {'Start':>8}  {'End':>8}  {'Frames':>7}  {'Gap to next cluster':>20}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")

    for i, cluster in enumerate(clusters):
        start = cluster[0][0]
        end = cluster[-1][0]
        count = len(cluster)
        gap_str = ""
        if i < len(clusters) - 1:
            next_start = clusters[i+1][0][0]
            gap_str = str(next_start - end)
        print(f"  {i+1:>3}  {start:>8}  {end:>8}  {count:>7}  {gap_str:>20}")

    # Print estimated episode lengths between boundaries
    print(f"\n  --- Estimated Episode Lengths (frames between boundary clusters) ---")
    if len(clusters) >= 2:
        # Before first boundary
        first_start = clusters[0][0][0]
        print(f"  Start -> Boundary 1: {first_start - 1} frames")

        for i in range(len(clusters) - 1):
            end_of_this = clusters[i][-1][0]
            start_of_next = clusters[i+1][0][0]
            length = start_of_next - end_of_this - 1
            print(f"  Boundary {i+1} -> Boundary {i+2}: {length} frames")

        # After last boundary
        last_end = clusters[-1][-1][0]
        if season == 1:
            total = 25462
        else:
            total = 19343
        print(f"  Boundary {len(clusters)} -> End: {total - last_end} frames")


def main():
    start_time = time.time()

    seasons = [
        (1, 25462),
        (2, 19343),
    ]

    for season, max_frame in seasons:
        print(f"\n{'#'*70}")
        print(f"# Scanning Season {season}: frames 1-{max_frame}")
        print(f"{'#'*70}")

        # Pass 1: Coarse scan (every 10th frame)
        t0 = time.time()
        print(f"\n  Pass 1: Coarse scan (every 10th frame)...")
        coarse_blacks = scan_frames(season, max_frame, step=10)
        t1 = time.time()
        coarse_time = t1 - t0
        print(f"  Coarse scan found {len(coarse_blacks)} black frames in {coarse_time:.1f}s")

        # Estimate full scan time
        est_full = coarse_time * 10
        print(f"  Estimated full scan time: {est_full:.0f}s")

        if est_full < 300:  # < 5 minutes, just do full scan
            print(f"\n  Pass 2: Full scan (every frame)...")
            t2 = time.time()
            all_blacks = scan_frames(season, max_frame, step=1)
            t3 = time.time()
            print(f"  Full scan completed in {t3-t2:.1f}s, found {len(all_blacks)} black frames")
            print_results(season, all_blacks)
        else:
            # Do fine scan around coarse hits only
            print(f"\n  Full scan would be too slow. Doing fine scan around coarse hits...")
            fine_frames = set()
            for fnum, val in coarse_blacks:
                for offset in range(-50, 51):
                    fine_frames.add(fnum + offset)

            fine_frames = sorted(f for f in fine_frames if 1 <= f <= max_frame)
            print(f"  Fine-scanning {len(fine_frames)} frames around {len(coarse_blacks)} coarse hits...")

            t2 = time.time()
            fine_blacks = scan_frames(season, max_frame, frame_range=fine_frames)
            t3 = time.time()
            print(f"  Fine scan completed in {t3-t2:.1f}s, found {len(fine_blacks)} black frames")
            print_results(season, fine_blacks)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Total execution time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
