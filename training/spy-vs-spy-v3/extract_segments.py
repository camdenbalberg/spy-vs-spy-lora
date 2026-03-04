#!/usr/bin/env python
"""
Phase 1: Detect and extract Spy vs Spy animated segments from MADtv episodes.

Spy vs Spy segments are black-and-white animated art embedded in full-color
live-action episodes. Detection uses HSV saturation analysis — animated
segments have near-zero saturation while live-action has high saturation.

Two-tier detection:
  - Core threshold (25): identifies definite greyscale animation frames
  - Expansion threshold (45): captures title cards and transitional frames
    adjacent to core regions

Episodes may contain multiple Spy vs Spy segments. All detected segments
are extracted and indexed (s1_e01_a.mkv, s1_e01_b.mkv, etc.).

Usage:
    python extract_segments.py                          # Process all episodes
    python extract_segments.py --episode "S01 E01"      # Process one episode
    python extract_segments.py --calibrate "S01 E01"    # Show saturation plot
    python extract_segments.py --dry-run                # Detect only, no extraction
    python extract_segments.py --threshold 25           # Override core threshold
    python extract_segments.py --padding 5              # Override padding seconds
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
VIDEOS_DIR = SCRIPT_DIR / "raw" / "videos"
OUTPUT_DIR = SCRIPT_DIR / "raw" / "episodes"

# === Detection parameters ===
DEFAULT_THRESHOLD = 25       # Core: definite greyscale (Spy vs Spy animation)
EXPANSION_THRESHOLD = 45     # Looser: catches title cards adjacent to core regions
SCAN_INTERVAL = 5            # Seconds between coarse scan samples
DEFAULT_PADDING = 5          # Seconds of padding on each side of extraction
MIN_SEGMENT_SEC = 10         # Minimum segment duration to keep
MAX_SEGMENT_SEC = 300        # Maximum segment duration
MERGE_GAP_SEC = 15           # Merge regions closer than this


def compute_frame_saturation(frame_bgr):
    """Compute mean saturation of the center 90% of a BGR frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    h, w = saturation.shape
    mh, mw = int(h * 0.05), int(w * 0.05)
    center = saturation[mh : h - mh, mw : w - mw]
    return float(np.mean(center))


def coarse_scan(video_path, interval_sec=SCAN_INTERVAL):
    """Sample one frame every interval_sec seconds using cv2."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    results = []
    t = 0.0
    while t < duration_sec:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        sat = compute_frame_saturation(frame)
        results.append((t, sat))
        t += interval_sec

    cap.release()
    return results, duration_sec


def find_greyscale_regions(scan_results, threshold=DEFAULT_THRESHOLD):
    """Find core greyscale regions, then expand boundaries using a looser threshold.

    Step 1: Find consecutive samples below the core threshold (25).
            Require at least 2 consecutive hits to form a core region.
    Step 2: Expand each core region's start/end to include adjacent samples
            below the expansion threshold (45). This captures title cards
            and transitional frames that border the animation.
    Step 3: Merge regions that are close together.
    """
    # Step 1: Find core regions (strict threshold)
    core_regions = []
    current_start = None
    current_end = None

    for timestamp, saturation in scan_results:
        if saturation < threshold:
            if current_start is None:
                current_start = timestamp
            current_end = timestamp
        else:
            if current_start is not None:
                if current_end > current_start:
                    core_regions.append((current_start, current_end))
                current_start = None
                current_end = None

    if current_start is not None and current_end is not None and current_end > current_start:
        core_regions.append((current_start, current_end))

    if not core_regions:
        return []

    # Step 2: Expand each region using the looser threshold
    # Build a lookup for fast timestamp -> index mapping
    timestamps = [t for t, _ in scan_results]
    saturations = [s for _, s in scan_results]

    expanded = []
    for core_start, core_end in core_regions:
        # Find the index of core_start and core_end in scan_results
        start_idx = timestamps.index(core_start)
        end_idx = timestamps.index(core_end)

        # Expand start backward
        while start_idx > 0 and saturations[start_idx - 1] < EXPANSION_THRESHOLD:
            start_idx -= 1

        # Expand end forward
        while end_idx < len(timestamps) - 1 and saturations[end_idx + 1] < EXPANSION_THRESHOLD:
            end_idx += 1

        expanded.append((timestamps[start_idx], timestamps[end_idx]))

    # Step 3: Merge close regions
    return merge_close_regions(expanded)


def merge_close_regions(regions, max_gap_sec=MERGE_GAP_SEC):
    """Merge regions separated by less than max_gap_sec."""
    if not regions:
        return regions
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] <= max_gap_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(r) for r in merged]


def refine_boundaries(video_path, region_start, region_end):
    """Refine region boundaries using FFmpeg frame extraction at 1-second intervals.

    Extracts frames from [region_start - 10, region_end + 10] at 1fps,
    analyzes saturation, and finds the precise first/last greyscale frame.
    Uses the expansion threshold to include title cards.
    """
    search_start = max(0, region_start - 10)
    search_end = region_end + 10
    search_duration = search_end - search_start

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{search_start:.3f}",
            "-i", str(video_path),
            "-t", f"{search_duration:.3f}",
            "-vf", "fps=1",
            "-q:v", "2",
            str(tmpdir / "frame_%04d.png"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return region_start, region_end

        frames = sorted(tmpdir.glob("frame_*.png"))
        if not frames:
            return region_start, region_end

        first_grey = None
        last_grey = None

        for i, f in enumerate(frames):
            t = search_start + i
            img = cv2.imread(str(f))
            if img is None:
                continue
            sat = compute_frame_saturation(img)
            if sat < EXPANSION_THRESHOLD:
                if first_grey is None:
                    first_grey = t
                last_grey = t

    return (first_grey if first_grey is not None else region_start,
            last_grey if last_grey is not None else region_end)


def filter_segments(regions):
    """Filter regions by duration. Return all valid segments."""
    valid = []
    for start, end in regions:
        duration = end - start
        if MIN_SEGMENT_SEC <= duration <= MAX_SEGMENT_SEC:
            valid.append((start, end))
    return valid


def extract_segment(input_path, output_path, start_sec, end_sec, padding_sec=DEFAULT_PADDING):
    """Extract segment using FFmpeg stream copy with padding."""
    padded_start = max(0, start_sec - padding_sec)
    duration = (end_sec + padding_sec) - padded_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{padded_start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "0:a:0",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-200:]}")
        return False
    return True


def save_thumbnail(video_path, timestamp_sec, thumb_path):
    """Extract a single frame as a JPEG thumbnail using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "5",
        str(thumb_path),
    ]
    subprocess.run(cmd, capture_output=True)


def parse_episode_id(filename):
    """'MADtv - S01 E01.mkv' -> ('s1', 'e01')"""
    match = re.search(r"S(\d+)\s*E(\d+)", filename)
    if not match:
        return None, None
    season = f"s{int(match.group(1))}"
    episode = f"e{match.group(2)}"
    return season, episode


def process_episode(video_path, season, ep, threshold, padding, dry_run):
    """Full pipeline for one episode. Returns list of manifest entries."""
    entries = []

    t0 = time.time()
    scan_results, duration = coarse_scan(video_path)
    scan_time = time.time() - t0

    if not scan_results:
        print(f"  FAILED: Could not read video")
        return [{"source": video_path.name, "status": "failed", "reason": "Could not read video"}]

    print(f"  Coarse scan: {len(scan_results)} frames in {scan_time:.1f}s")

    regions = find_greyscale_regions(scan_results, threshold=threshold)
    print(f"  Found {len(regions)} greyscale region(s) after expansion+merge")

    # Log all regions
    all_regions_info = []
    for i, (rs, re_) in enumerate(regions):
        rdur = re_ - rs
        all_regions_info.append({"start": round(rs, 1), "end": round(re_, 1), "duration": round(rdur, 1)})
        tag = ""
        if rdur < MIN_SEGMENT_SEC:
            tag = " -- FILTERED (too short)"
        elif rdur > MAX_SEGMENT_SEC:
            tag = " -- FILTERED (too long)"
        else:
            tag = " -- CANDIDATE"
        print(f"    Region {i + 1}: {rs:.1f}s - {re_:.1f}s ({rdur:.1f}s){tag}")

    segments = filter_segments(regions)

    if not segments:
        print(f"  MANUAL REVIEW: No valid segments found")
        return [{
            "source": video_path.name,
            "output": None,
            "status": "manual_review",
            "reason": f"{len(regions)} region(s), none in valid duration range ({MIN_SEGMENT_SEC}-{MAX_SEGMENT_SEC}s)",
            "candidates": all_regions_info,
        }]

    # Refine and extract each segment
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        suffix = f"_{chr(ord('a') + seg_idx)}" if len(segments) > 1 else ""
        output_name = f"{season}_{ep}{suffix}.mkv"
        output_path = OUTPUT_DIR / output_name

        print(f"  Refining segment {seg_idx + 1}/{len(segments)}...", end=" ", flush=True)
        refined_start, refined_end = refine_boundaries(video_path, seg_start, seg_end)
        refined_dur = refined_end - refined_start
        print(f"{refined_start:.1f}s - {refined_end:.1f}s ({refined_dur:.1f}s)")

        entry = {
            "source": video_path.name,
            "output": output_name,
            "segment_index": seg_idx,
            "total_segments": len(segments),
            "detected_start": round(refined_start, 1),
            "detected_end": round(refined_end, 1),
            "detected_duration": round(refined_dur, 1),
            "padded_start": round(max(0, refined_start - padding), 1),
            "padded_end": round(refined_end + padding, 1),
            "padded_duration": round((refined_end + padding) - max(0, refined_start - padding), 1),
        }

        if dry_run:
            entry["status"] = "dry_run"
            print(f"    DRY RUN: Would extract to {output_name}")
        else:
            print(f"    Extracting {output_name} ({padding}s padding)...", end=" ", flush=True)
            t0 = time.time()
            success = extract_segment(video_path, output_path, refined_start, refined_end, padding)
            ext_time = time.time() - t0
            if success:
                entry["status"] = "ok"
                print(f"done ({ext_time:.1f}s)")
                # Save thumbnail from the middle of the segment
                thumb_path = OUTPUT_DIR / output_name.replace(".mkv", ".jpg")
                mid_time = (refined_start + refined_end) / 2
                save_thumbnail(video_path, mid_time, thumb_path)
            else:
                entry["status"] = "extraction_failed"
                print("FAILED")

        entries.append(entry)

    return entries


def calibrate_episode(video_path, threshold):
    """Print a saturation timeline with ASCII bar chart for one episode."""
    print(f"\nCalibrating on: {video_path.name}")
    print(f"Scanning at {SCAN_INTERVAL}-second intervals...\n")

    scan_results, duration = coarse_scan(video_path)
    if not scan_results:
        print("ERROR: Could not read video")
        return

    max_sat = max(s for _, s in scan_results) if scan_results else 1
    bar_width = 50

    print(f"{'Time':>8s}  {'Sat':>5s}  Bar")
    print(f"{'----':>8s}  {'---':>5s}  ---")

    for timestamp, saturation in scan_results:
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        bar_len = int((saturation / max(max_sat, 1)) * bar_width)
        bar = "=" * bar_len
        marker = ""
        if saturation < threshold:
            marker = "  << CORE"
        elif saturation < EXPANSION_THRESHOLD:
            marker = "  < edge"
        print(f"{minutes:3d}:{seconds:02d}  {saturation:5.1f}  {bar}{marker}")

    # Run full detection
    regions = find_greyscale_regions(scan_results, threshold=threshold)
    print(f"\nDetected {len(regions)} region(s) after expansion+merge:")
    for i, (rs, re_) in enumerate(regions):
        rdur = re_ - rs
        m1, s1 = int(rs // 60), int(rs % 60)
        m2, s2 = int(re_ // 60), int(re_ % 60)
        tag = "VALID" if MIN_SEGMENT_SEC <= rdur <= MAX_SEGMENT_SEC else "filtered"
        print(f"  Region {i + 1}: {m1}:{s1:02d} - {m2}:{s2:02d} ({rdur:.0f}s) [{tag}]")

    segments = filter_segments(regions)
    if segments:
        print(f"\nRefining {len(segments)} segment(s)...")
        for seg_idx, (ss, se) in enumerate(segments):
            rs, re_ = refine_boundaries(video_path, ss, se)
            rdur = re_ - rs
            m1, s1 = int(rs // 60), int(rs % 60)
            m2, s2 = int(re_ // 60), int(re_ % 60)
            print(f"  Segment {seg_idx + 1}: {m1}:{s1:02d} - {m2}:{s2:02d} ({rdur:.0f}s)")

    # Stats
    greyscale = [s for _, s in scan_results if s < threshold]
    edge = [s for _, s in scan_results if threshold <= s < EXPANSION_THRESHOLD]
    color = [s for _, s in scan_results if s >= EXPANSION_THRESHOLD]
    print(f"\nSaturation distribution:")
    print(f"  Core (<{threshold}):       {len(greyscale)} samples"
          + (f", mean={np.mean(greyscale):.1f}, max={max(greyscale):.1f}" if greyscale else ""))
    print(f"  Edge ({threshold}-{EXPANSION_THRESHOLD}):    {len(edge)} samples"
          + (f", mean={np.mean(edge):.1f}" if edge else ""))
    print(f"  Color (>={EXPANSION_THRESHOLD}):      {len(color)} samples"
          + (f", mean={np.mean(color):.1f}, min={min(color):.1f}" if color else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", help='Process single episode, e.g. "S01 E01"')
    parser.add_argument("--calibrate", help='Show saturation plot for one episode, e.g. "S01 E01"')
    parser.add_argument("--dry-run", action="store_true", help="Detect only, no extraction")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Core saturation threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                        help=f"Padding in seconds (default: {DEFAULT_PADDING})")
    args = parser.parse_args()

    if not VIDEOS_DIR.exists():
        print(f"ERROR: Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)

    # Calibrate mode
    if args.calibrate:
        matches = sorted(VIDEOS_DIR.glob(f"*{args.calibrate}*"))
        if not matches:
            print(f"ERROR: No episode matching '{args.calibrate}' found in {VIDEOS_DIR}")
            sys.exit(1)
        calibrate_episode(matches[0], args.threshold)
        return

    # Batch or single processing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEOS_DIR.glob("MADtv - *.mkv"))
    if not videos:
        print(f"ERROR: No MKV files found in {VIDEOS_DIR}")
        sys.exit(1)

    if args.episode:
        videos = [v for v in videos if args.episode in v.name]
        if not videos:
            print(f"ERROR: No episode matching '{args.episode}'")
            sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "EXTRACT"
    print(f"Phase 1: Detecting Spy vs Spy segments [{mode}]")
    print(f"Episodes: {len(videos)} | Core threshold: {args.threshold} | "
          f"Expansion: {EXPANSION_THRESHOLD} | Padding: {args.padding}s")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    manifest = {
        "generated": datetime.now().isoformat(),
        "threshold": args.threshold,
        "expansion_threshold": EXPANSION_THRESHOLD,
        "padding_sec": args.padding,
        "scan_interval_sec": SCAN_INTERVAL,
        "episodes": [],
    }

    counts = {"ok": 0, "dry_run": 0, "manual_review": 0, "failed": 0, "extraction_failed": 0}

    for i, video in enumerate(videos):
        print(f"\n[{i + 1:3d}/{len(videos)}] {video.name}")

        season, ep = parse_episode_id(video.name)
        if season is None:
            print(f"  SKIPPED: Could not parse episode ID from filename")
            continue

        entries = process_episode(video, season, ep, args.threshold, args.padding, args.dry_run)
        manifest["episodes"].extend(entries)
        for e in entries:
            status = e.get("status", "failed")
            counts[status] = counts.get(status, 0) + 1

    # Summary
    manifest["summary"] = {
        "total_episodes": len(videos),
        "total_segments": counts["ok"] + counts["dry_run"],
        "extracted": counts["ok"],
        "dry_run": counts["dry_run"],
        "manual_review": counts["manual_review"],
        "failed": counts["failed"] + counts["extraction_failed"],
    }

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results: {counts['ok']} extracted, {counts['dry_run']} dry-run, "
          f"{counts['manual_review']} manual review, {counts['failed'] + counts['extraction_failed']} failed")
    print(f"Manifest: {manifest_path}")

    review = [e for e in manifest["episodes"] if e.get("status") == "manual_review"]
    if review:
        print(f"\nManual review needed:")
        for e in review:
            print(f"  - {e['source']}: {e.get('reason', 'unknown')}")


if __name__ == "__main__":
    main()
