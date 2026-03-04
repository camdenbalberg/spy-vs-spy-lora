#!/usr/bin/env python3
"""spy_factory.py - Automated LoRA training data generation from Spy vs Spy episodes.

Parses Gemini caption files, extracts the sharpest frame at each timestamp
using FFmpeg + OpenCV, appends LoRA training tags, and saves paired .jpg + .txt files.

v4 changes:
- Three trigger words: spyvspy (style), white_spy, black_spy (character)
- No style suffix
- nonstandard_outfit support (skips appearance suffix)
- Updated appearance suffix logic for mixed hat states
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

# ── Path Discovery ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
EPISODES_DIR = PROJECT_ROOT / "raw" / "gemini"
CAPTIONS_DIR = PROJECT_ROOT / "captions"
FRAMES_DIR = PROJECT_ROOT / "frames"
DATASET_DIR = PROJECT_ROOT / "dataset"

# ── Constants ───────────────────────────────────────────────────────────────────

TRIGGER_STYLE = "spyvspy"
TRIGGER_WHITE = "white_spy"
TRIGGER_BLACK = "black_spy"

APPEARANCE_SINGULAR = (
    "wearing a fedora hat and trenchcoat with long pointed beak nose "
    "and black sclera eyes"
)
APPEARANCE_SINGULAR_HATLESS = (
    "hatless, wearing trenchcoat with long pointed beak nose "
    "and black sclera eyes"
)
APPEARANCE_PLURAL = (
    "both wearing fedora hats and trenchcoats with long pointed beak noses "
    "and black sclera eyes"
)
APPEARANCE_PLURAL_BOTH_HATLESS = (
    "both hatless, wearing trenchcoats with long pointed beak noses "
    "and black sclera eyes"
)


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [CAPTIONS_DIR, FRAMES_DIR, DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_episode_video(stem):
    """Find a video file matching the given stem, regardless of extension."""
    if not EPISODES_DIR.exists():
        return None
    for ext in (".mp4", ".mkv", ".avi", ".webm"):
        candidate = EPISODES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: search all files in episodes dir
    for f in EPISODES_DIR.iterdir():
        if f.stem == stem and f.is_file() and f.suffix.lower() not in (".txt", ".md"):
            return f
    return None


# ── Caption Parsing ─────────────────────────────────────────────────────────────


def _detect_episode_header(line):
    """Try to match an episode header line. Returns episode stem or None."""
    # Pattern: ### EPISODE: filename.mp4
    m = re.match(r"^#{1,4}\s*EPISODE:\s*(.+?)\s*$", line)
    if m:
        return Path(m.group(1).strip()).stem

    # Pattern: === EPISODE: filename.mp4 ===
    m = re.match(r"^={3,}\s*EPISODE:\s*(.+?)\s*={0,}\s*$", line)
    if m:
        return Path(m.group(1).strip()).stem

    # Pattern: CSV row "EPISODE: filename.mp4,,"
    m = re.match(r"^EPISODE:\s*(.+?)(?:,,.*)?$", line)
    if m:
        return Path(m.group(1).strip()).stem

    # Pattern: Markdown table row "| EPISODE: filename.mp4 |"
    m = re.match(r"^\|\s*EPISODE:\s*(.+?)\s*\|", line)
    if m:
        return Path(m.group(1).strip()).stem

    return None


def _parse_data_row(line):
    """Try to parse a data row. Returns (timestamp, spy, caption) or None."""
    # Markdown table: | MM:SS | Spy(s) | Description |
    m = re.match(r"^\|\s*(\d+:\d+)\s*\|\s*(\w+)\s*\|\s*(.+?)\s*\|?\s*$", line)
    if m:
        return m.group(1), m.group(2).strip(), m.group(3).strip()

    # CSV row
    try:
        reader = csv.reader([line])
        row = next(reader)
        if len(row) >= 3 and re.match(r"^\d+:\d+$", row[0].strip()):
            return row[0].strip(), row[1].strip(), row[2].strip()
    except (csv.Error, StopIteration):
        pass

    return None


def parse_caption_file(filepath):
    """Parse a caption file (CSV or markdown table format).

    Returns dict mapping episode_stem -> list of (timestamp_str, spy, caption).
    """
    episodes = {}
    current_episode = None

    text = Path(filepath).read_text(encoding="utf-8")
    lines = text.strip().splitlines()

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check for episode header
        ep_stem = _detect_episode_header(line_stripped)
        if ep_stem:
            current_episode = ep_stem
            if current_episode not in episodes:
                episodes[current_episode] = []
            continue

        if current_episode is None:
            continue

        # Skip markdown table headers and separators
        if re.match(r"^\|\s*(Timestamp|---)", line_stripped):
            continue
        if re.match(r"^Timestamp\s*\(", line_stripped, re.IGNORECASE):
            continue

        parsed = _parse_data_row(line_stripped)
        if parsed is None:
            continue

        ts, spy, caption = parsed

        # Skip SKIP and DUPLICATE rows
        if caption.upper().startswith("SKIP") or caption.upper().startswith("DUPLICATE"):
            continue

        episodes[current_episode].append((ts, spy, caption))

    return episodes


def parse_all_captions():
    """Parse all caption files in the captions directory.

    Returns dict mapping episode_stem -> list of (timestamp_str, spy, caption).
    """
    all_episodes = {}

    if not CAPTIONS_DIR.exists():
        return all_episodes

    for f in sorted(CAPTIONS_DIR.iterdir()):
        if f.suffix.lower() in (".csv", ".txt", ".md"):
            episodes = parse_caption_file(f)
            for ep, rows in episodes.items():
                all_episodes[ep] = rows

    return all_episodes


# ── Timestamp Helpers ───────────────────────────────────────────────────────────


def timestamp_to_seconds(ts):
    """Convert MM:SS or H:MM:SS timestamp to float seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_timestamp(sec):
    """Convert seconds to M:SS format."""
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


# ── Frame Extraction ────────────────────────────────────────────────────────────


def find_selected_frame(episode_dir, shot_prefix):
    """Find the _selected frame for a shot via glob. Returns Path or None."""
    matches = list(episode_dir.glob(f"{shot_prefix}_*_selected.jpg"))
    return matches[0] if matches else None


def extract_frames_for_shot(video_path, start_sec, end_sec, episode_dir, shot_num):
    """Extract frames for a single shot using FFmpeg.

    Returns list of extracted frame paths, or empty list on failure.
    """
    shot_prefix = f"s{shot_num:02d}"
    temp_pattern = str(episode_dir / f"_temp_{shot_prefix}_%04d.jpg")

    # Try with mpdecimate first (drops duplicate/freeze frames)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(video_path),
        "-vf", "yadif,mpdecimate,setpts=N/FRAME_RATE/TB,hue=s=0",
        "-qscale:v", "2",
        temp_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    temp_frames = sorted(episode_dir.glob(f"_temp_{shot_prefix}_*.jpg"))

    if not temp_frames:
        # Fallback: extract without mpdecimate
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", str(video_path),
            "-vf", "yadif,hue=s=0",
            "-qscale:v", "2",
            temp_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        temp_frames = sorted(episode_dir.glob(f"_temp_{shot_prefix}_*.jpg"))

    if not temp_frames:
        print(f"  WARNING: No frames extracted for {shot_prefix} "
              f"({start_sec}s-{end_sec}s)")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-3:]:
                print(f"    {line}")
        return []

    # Rename sequentially: s01_01.jpg, s01_02.jpg, ...
    renamed = []
    for i, temp_path in enumerate(temp_frames, start=1):
        new_name = episode_dir / f"{shot_prefix}_{i:02d}.jpg"
        temp_path.rename(new_name)
        renamed.append(new_name)

    return renamed


def process_episode(episode_stem, rows):
    """Process a single episode: extract frames, score, build manifest.

    Returns True if episode was processed (or already done), False on error.
    """
    episode_dir = FRAMES_DIR / episode_stem

    # Resumability: skip if already extracted
    if episode_dir.exists() and find_selected_frame(episode_dir, "s01"):
        print(f"  SKIP {episode_stem} (already extracted)")
        return True

    video_path = find_episode_video(episode_stem)
    if video_path is None:
        print(f"  ERROR: No video file found for {episode_stem}")
        return False

    print(f"  Processing {episode_stem} "
          f"({len(rows)} shots from {video_path.name})")
    episode_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"episode": f"{episode_stem}{video_path.suffix}", "shots": []}

    for shot_idx, (ts, spy, caption) in enumerate(rows):
        shot_num = shot_idx + 1
        start_sec = timestamp_to_seconds(ts)

        # End time: next caption's timestamp, or +2s for last shot
        if shot_idx + 1 < len(rows):
            end_sec = timestamp_to_seconds(rows[shot_idx + 1][0])
        else:
            end_sec = start_sec + 2

        # Ensure window is at least 0.5s
        if end_sec <= start_sec:
            end_sec = start_sec + 1

        shot_prefix = f"s{shot_num:02d}"
        print(f"    Shot {shot_num:02d} [{ts} -> {seconds_to_timestamp(end_sec)}] "
              f"({spy})", end="", flush=True)

        frames = extract_frames_for_shot(
            video_path, start_sec, end_sec, episode_dir, shot_num
        )

        if not frames:
            print(" - 0 frames")
            manifest["shots"].append({
                "shot": shot_num,
                "timestamp_start": ts,
                "timestamp_end": seconds_to_timestamp(end_sec),
                "spy": spy,
                "spy_original": None,
                "caption": caption,
                "candidates": 0,
                "selected": None,
                "excluded": False,
                "white_hat_off": False,
                "black_hat_off": False,
                "nonstandard_outfit": False,
                "appearance_suffix": None,
            })
            continue

        # Select midpoint frame and rename it in-place
        mid = len(frames) // 2
        selected_frame = frames[mid]
        m = re.match(rf"{shot_prefix}_(\d+)\.jpg", selected_frame.name)
        mid_idx = m.group(1) if m else f"{mid + 1:02d}"

        renamed = selected_frame.with_name(
            f"{shot_prefix}_{mid_idx}_selected.jpg"
        )
        selected_frame.rename(renamed)

        # Save raw caption
        caption_path = episode_dir / f"{shot_prefix}_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")

        print(f" - {len(frames)} frames, mid={mid_idx}")

        manifest["shots"].append({
            "shot": shot_num,
            "timestamp_start": ts,
            "timestamp_end": seconds_to_timestamp(end_sec),
            "spy": spy,
            "spy_original": None,
            "caption": caption,
            "candidates": len(frames),
            "selected": mid_idx,
            "excluded": False,
            "white_hat_off": False,
            "black_hat_off": False,
            "nonstandard_outfit": False,
            "appearance_suffix": None,
        })

    # Write manifest
    manifest_path = episode_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"  Done: {len(manifest['shots'])} shots, manifest written")
    return True


# ── Dataset Building ────────────────────────────────────────────────────────────


def build_trigger_prefix(spy):
    """Build the trigger word prefix based on spy type.

    Returns string like 'spyvspy, white_spy, ' or 'spyvspy, white_spy, black_spy, '.
    """
    spy_lower = (spy or "None").lower()
    if spy_lower == "white":
        return f"{TRIGGER_STYLE}, {TRIGGER_WHITE}, "
    elif spy_lower == "black":
        return f"{TRIGGER_STYLE}, {TRIGGER_BLACK}, "
    elif spy_lower == "both":
        return f"{TRIGGER_STYLE}, {TRIGGER_WHITE}, {TRIGGER_BLACK}, "
    else:
        return f"{TRIGGER_STYLE}, "


def compute_appearance_suffix(spy, white_hat_off=False, black_hat_off=False,
                              nonstandard_outfit=False):
    """Compute the default appearance suffix based on spy type and hat flags.

    v4: No longer inspects caption text. Uses spy type directly.
    Returns the suffix string (without leading comma), or empty string.
    """
    if nonstandard_outfit:
        return ""

    spy_lower = (spy or "").lower()

    if spy_lower in ("white", "black"):
        hat_off = white_hat_off if spy_lower == "white" else black_hat_off
        return APPEARANCE_SINGULAR_HATLESS if hat_off else APPEARANCE_SINGULAR

    elif spy_lower == "both":
        if white_hat_off and black_hat_off:
            return APPEARANCE_PLURAL_BOTH_HATLESS
        elif not white_hat_off and not black_hat_off:
            return APPEARANCE_PLURAL
        else:
            # Mixed: one hat off, one hat on
            w = "hatless" if white_hat_off else "wearing fedora hat"
            b = "hatless" if black_hat_off else "wearing fedora hat"
            return (
                f"white spy {w}, black spy {b}, "
                f"both wearing trenchcoats with long pointed beak noses "
                f"and black sclera eyes"
            )

    # None — no appearance suffix
    return ""


def build_caption_text(raw_caption, spy, appearance_suffix=None,
                       nonstandard_outfit=False):
    """Build final caption with trigger words and appearance suffix.

    v4: No style suffix. Character triggers based on spy type.
    """
    prefix = build_trigger_prefix(spy)
    text = f"{prefix}{raw_caption}"

    if appearance_suffix is None:
        appearance_suffix = compute_appearance_suffix(spy, nonstandard_outfit=nonstandard_outfit)

    if appearance_suffix:
        text += f", {appearance_suffix}"

    return text


def build_dataset_for_episode(episode_stem):
    """Build dataset files for a single episode from its frames/ data.

    Returns count of pairs written.
    """
    episode_dir = FRAMES_DIR / episode_stem
    manifest_path = episode_dir / "manifest.json"

    if not manifest_path.exists():
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = 0

    for shot_info in manifest["shots"]:
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        ds_img = DATASET_DIR / f"{episode_stem}_{shot_prefix}.jpg"
        ds_txt = DATASET_DIR / f"{episode_stem}_{shot_prefix}.txt"

        if shot_info.get("excluded", False):
            ds_img.unlink(missing_ok=True)
            ds_txt.unlink(missing_ok=True)
            continue

        selected_path = find_selected_frame(episode_dir, shot_prefix)
        if not selected_path:
            continue

        # Read current caption from caption.txt (may have been edited)
        caption_path = episode_dir / f"{shot_prefix}_caption.txt"
        if caption_path.exists():
            raw_caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            raw_caption = shot_info.get("caption", "")

        spy = shot_info.get("spy", "None")
        appearance_suffix = shot_info.get("appearance_suffix")
        nonstandard_outfit = shot_info.get("nonstandard_outfit", False)

        shutil.copy2(selected_path, ds_img)
        ds_txt.write_text(
            build_caption_text(
                raw_caption, spy,
                appearance_suffix=appearance_suffix,
                nonstandard_outfit=nonstandard_outfit,
            ),
            encoding="utf-8",
        )
        count += 1

    return count


# ── CLI Commands ────────────────────────────────────────────────────────────────


def cmd_caption(_args):
    """List caption files and stats."""
    ensure_dirs()

    caption_files = sorted(
        f for f in CAPTIONS_DIR.iterdir()
        if f.suffix.lower() in (".csv", ".txt", ".md")
    ) if CAPTIONS_DIR.exists() else []

    if not caption_files:
        print(f"No caption files found in {CAPTIONS_DIR}")
        print("Save Gemini caption output as .csv, .txt, or .md files there.")
        return

    print(f"Caption files in {CAPTIONS_DIR}:\n")
    total_episodes = 0
    total_rows = 0

    for f in caption_files:
        episodes = parse_caption_file(f)
        ep_count = len(episodes)
        row_count = sum(len(rows) for rows in episodes.values())
        total_episodes += ep_count
        total_rows += row_count
        print(f"  {f.name}: {ep_count} episodes, {row_count} caption rows")
        for ep_name in sorted(episodes.keys()):
            print(f"    {ep_name}: {len(episodes[ep_name])} rows")

    print(f"\nTotal: {len(caption_files)} files, "
          f"{total_episodes} episodes, {total_rows} rows")


def cmd_extract(args):
    """Extract frames and build dataset."""
    ensure_dirs()

    episode_filter = getattr(args, "episode", None)
    all_episodes = parse_all_captions()

    if not all_episodes:
        print("No captions found. Save caption files in the captions/ directory.")
        return

    # Resolve episode filter (exact match, then prefix match)
    if episode_filter and episode_filter not in all_episodes:
        matches = [k for k in all_episodes if k.startswith(episode_filter)]
        if len(matches) == 1:
            episode_filter = matches[0]
        elif matches:
            print(f"Ambiguous filter '{episode_filter}', matches: {matches}")
            return
        else:
            print(f"Episode '{episode_filter}' not found in captions.")
            return

    print("=== FRAME EXTRACTION ===\n")

    extracted = 0
    errors = 0

    for episode_stem in sorted(all_episodes.keys()):
        if episode_filter and episode_stem != episode_filter:
            continue
        rows = all_episodes[episode_stem]
        if process_episode(episode_stem, rows):
            extracted += 1
        else:
            errors += 1

    print(f"\nExtraction: {extracted} episodes processed, {errors} errors")

    # Always rebuild dataset
    print("\n=== BUILDING DATASET ===\n")

    total_ds = 0
    for episode_stem in sorted(all_episodes.keys()):
        if episode_filter and episode_stem != episode_filter:
            continue
        count = build_dataset_for_episode(episode_stem)
        if count > 0:
            print(f"  {episode_stem}: {count} pairs")
        total_ds += count

    print(f"\nDataset: {total_ds} image-caption pairs in {DATASET_DIR}")


def _parse_shot_arg(shot_str):
    """Parse a shot argument like 's03' or '03' into an int."""
    return int(shot_str.lstrip("s").lstrip("0") or "0")


def cmd_swap(args):
    """Swap a shot's selected frame to a different candidate."""
    episode_stem = args.episode
    shot_num = _parse_shot_arg(args.shot)
    candidate = args.candidate
    shot_prefix = f"s{shot_num:02d}"

    episode_dir = FRAMES_DIR / episode_stem
    manifest_path = episode_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest found for {episode_stem}")
        return

    candidate_path = episode_dir / f"{shot_prefix}_{candidate}.jpg"
    if not candidate_path.exists():
        print(f"ERROR: Candidate frame {candidate_path.name} not found")
        return

    # Un-rename the current selection back to a plain candidate
    old_selected = find_selected_frame(episode_dir, shot_prefix)
    if old_selected:
        # e.g. s03_05_selected.jpg -> s03_05.jpg
        old_plain = old_selected.with_name(
            old_selected.name.replace("_selected", "")
        )
        old_selected.rename(old_plain)

    # Rename the new candidate as selected
    new_selected = candidate_path.with_name(
        f"{shot_prefix}_{candidate}_selected.jpg"
    )
    candidate_path.rename(new_selected)

    # Update manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shot_info in manifest["shots"]:
        if shot_info["shot"] == shot_num:
            shot_info["selected"] = candidate
            break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update dataset image
    ds_img = DATASET_DIR / f"{episode_stem}_{shot_prefix}.jpg"
    if DATASET_DIR.exists():
        shutil.copy2(new_selected, ds_img)

    print(f"Swapped {episode_stem} shot {shot_prefix} -> candidate {candidate}")


def cmd_exclude(args):
    """Exclude a shot from the dataset."""
    episode_stem = args.episode
    shot_num = _parse_shot_arg(args.shot)
    shot_prefix = f"s{shot_num:02d}"

    episode_dir = FRAMES_DIR / episode_stem
    manifest_path = episode_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest found for {episode_stem}")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shot_info in manifest["shots"]:
        if shot_info["shot"] == shot_num:
            shot_info["excluded"] = True
            break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ds_img = DATASET_DIR / f"{episode_stem}_{shot_prefix}.jpg"
    ds_txt = DATASET_DIR / f"{episode_stem}_{shot_prefix}.txt"
    ds_img.unlink(missing_ok=True)
    ds_txt.unlink(missing_ok=True)

    print(f"Excluded {episode_stem} shot {shot_prefix}")


def cmd_include(args):
    """Undo exclusion of a shot."""
    episode_stem = args.episode
    shot_num = _parse_shot_arg(args.shot)
    shot_prefix = f"s{shot_num:02d}"

    episode_dir = FRAMES_DIR / episode_stem
    manifest_path = episode_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest found for {episode_stem}")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spy = "None"
    raw_caption = ""
    nonstandard_outfit = False
    for shot_info in manifest["shots"]:
        if shot_info["shot"] == shot_num:
            shot_info["excluded"] = False
            spy = shot_info.get("spy", "None")
            raw_caption = shot_info.get("caption", "")
            nonstandard_outfit = shot_info.get("nonstandard_outfit", False)
            break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Restore dataset entry
    selected_path = find_selected_frame(episode_dir, shot_prefix)
    caption_path = episode_dir / f"{shot_prefix}_caption.txt"

    if selected_path:
        ds_img = DATASET_DIR / f"{episode_stem}_{shot_prefix}.jpg"
        shutil.copy2(selected_path, ds_img)

    if caption_path.exists():
        raw_caption = caption_path.read_text(encoding="utf-8").strip()

    ds_txt = DATASET_DIR / f"{episode_stem}_{shot_prefix}.txt"
    ds_txt.write_text(
        build_caption_text(raw_caption, spy, nonstandard_outfit=nonstandard_outfit),
        encoding="utf-8",
    )

    print(f"Included {episode_stem} shot {shot_prefix}")


def cmd_edit_caption(args):
    """Edit a shot's caption."""
    episode_stem = args.episode
    shot_num = _parse_shot_arg(args.shot)
    new_caption = args.caption
    shot_prefix = f"s{shot_num:02d}"

    episode_dir = FRAMES_DIR / episode_stem
    manifest_path = episode_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest found for {episode_stem}")
        return

    # Read old caption
    caption_path = episode_dir / f"{shot_prefix}_caption.txt"
    old_caption = ""
    if caption_path.exists():
        old_caption = caption_path.read_text(encoding="utf-8").strip()

    # Write new caption
    caption_path.write_text(new_caption, encoding="utf-8")

    # Update manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spy = "None"
    excluded = False
    nonstandard_outfit = False
    for shot_info in manifest["shots"]:
        if shot_info["shot"] == shot_num:
            shot_info["caption"] = new_caption
            spy = shot_info.get("spy", "None")
            excluded = shot_info.get("excluded", False)
            nonstandard_outfit = shot_info.get("nonstandard_outfit", False)
            break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Regenerate dataset caption (unless excluded)
    if not excluded:
        ds_txt = DATASET_DIR / f"{episode_stem}_{shot_prefix}.txt"
        ds_txt.write_text(
            build_caption_text(new_caption, spy, nonstandard_outfit=nonstandard_outfit),
            encoding="utf-8",
        )

    old_display = old_caption[:80] + ("..." if len(old_caption) > 80 else "")
    new_display = new_caption[:80] + ("..." if len(new_caption) > 80 else "")
    print(f"Caption updated for {episode_stem} shot {shot_prefix}:")
    print(f"  OLD: {old_display}")
    print(f"  NEW: {new_display}")


def cmd_rebuild_dataset(_args):
    """Rebuild all dataset files from frames/ data."""
    ensure_dirs()

    if not FRAMES_DIR.exists():
        print("No frames directory found. Run extract first.")
        return

    total = 0
    for episode_dir in sorted(FRAMES_DIR.iterdir()):
        if not episode_dir.is_dir():
            continue
        count = build_dataset_for_episode(episode_dir.name)
        if count > 0:
            print(f"  {episode_dir.name}: {count} pairs")
        total += count

    print(f"\nDataset rebuilt: {total} image-caption pairs in {DATASET_DIR}")


def cmd_summary(_args):
    """Print dataset summary statistics."""
    ensure_dirs()

    # ── Caption stats ───────────────────────────────────────────────────────
    caption_files = sorted(
        f for f in CAPTIONS_DIR.iterdir()
        if f.suffix.lower() in (".csv", ".txt", ".md")
    ) if CAPTIONS_DIR.exists() else []

    all_episode_rows = {}  # episode_stem -> valid row count
    total_rows = 0
    total_skipped = 0

    for f in caption_files:
        text = f.read_text(encoding="utf-8")
        current_ep = None

        for line in text.strip().splitlines():
            line_s = line.strip()
            if not line_s:
                continue

            ep_stem = _detect_episode_header(line_s)
            if ep_stem:
                current_ep = ep_stem
                all_episode_rows.setdefault(current_ep, 0)
                continue

            if current_ep is None:
                continue
            if re.match(r"^\|\s*(Timestamp|---)", line_s):
                continue
            if re.match(r"^Timestamp\s*\(", line_s, re.IGNORECASE):
                continue

            parsed = _parse_data_row(line_s)
            if parsed is None:
                continue

            _, _, caption_text = parsed
            total_rows += 1
            if (caption_text.upper().startswith("SKIP")
                    or caption_text.upper().startswith("DUPLICATE")):
                total_skipped += 1
            else:
                all_episode_rows[current_ep] = (
                    all_episode_rows.get(current_ep, 0) + 1
                )

    valid_captions = total_rows - total_skipped

    # ── Frames stats ────────────────────────────────────────────────────────
    episodes_with_frames = 0
    total_candidates = 0
    total_shots = 0

    if FRAMES_DIR.exists():
        for ep_dir in sorted(FRAMES_DIR.iterdir()):
            if not ep_dir.is_dir():
                continue
            manifest_path = ep_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            episodes_with_frames += 1
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for shot in manifest.get("shots", []):
                total_shots += 1
                total_candidates += shot.get("candidates", 0)

    avg_candidates = total_candidates / total_shots if total_shots > 0 else 0

    # ── Dataset stats ───────────────────────────────────────────────────────
    ds_image_stems = set()
    ds_caption_stems = set()

    if DATASET_DIR.exists():
        for f in DATASET_DIR.iterdir():
            if f.suffix == ".jpg":
                ds_image_stems.add(f.stem)
            elif f.suffix == ".txt":
                ds_caption_stems.add(f.stem)

    mismatches = len(ds_image_stems.symmetric_difference(ds_caption_stems))

    missing_videos = sum(
        1 for ep in all_episode_rows if find_episode_video(ep) is None
    )

    print("=== SPY VS SPY DATASET SUMMARY (v4) ===")
    print(f"Caption files found: {len(caption_files)}")
    print(f"Episodes in captions: {len(all_episode_rows)}")
    print(f"Total caption rows: {total_rows}")
    print(f"Rows skipped (SKIP/DUPLICATE): {total_skipped}")
    print(f"Valid captions: {valid_captions}")
    print()
    print("Review folder:")
    print(f"  Episodes with frames: {episodes_with_frames}")
    print(f"  Total candidate frames: ~{total_candidates:,}")
    print(f"  Frames per shot (avg): ~{avg_candidates:.0f}")
    print()
    print("Dataset folder:")
    print(f"  Image files: {len(ds_image_stems)}")
    print(f"  Caption files: {len(ds_caption_stems)}")
    print(f"  Mismatches (jpg without txt or vice versa): {mismatches}")
    print(f"  Episodes missing video file: {missing_videos}")


def cmd_run(args):
    """Run extract then summary."""
    cmd_extract(args)
    print()
    cmd_summary(args)


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Spy vs Spy LoRA training data pipeline (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands:
  caption          List caption files and stats
  extract          Extract frames and build dataset
  swap             Swap a shot's selected frame
  exclude          Exclude a shot from dataset
  include          Undo shot exclusion
  edit-caption     Edit a shot's caption
  rebuild-dataset  Regenerate all dataset files from frames
  summary          Print dataset statistics
  run              Extract then summary
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # caption
    subparsers.add_parser("caption", help="List caption files and stats")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract frames + build dataset")
    p_extract.add_argument(
        "--episode", type=str, default=None, help="Single episode stem to process"
    )

    # swap
    p_swap = subparsers.add_parser("swap", help="Swap selected frame for a shot")
    p_swap.add_argument("episode", help="Episode stem (e.g. s1_e01_a)")
    p_swap.add_argument("shot", help="Shot number (e.g. s03 or 03)")
    p_swap.add_argument("candidate", help="Candidate frame number (e.g. 02)")

    # exclude
    p_exclude = subparsers.add_parser("exclude", help="Exclude a shot from dataset")
    p_exclude.add_argument("episode", help="Episode stem")
    p_exclude.add_argument("shot", help="Shot number")

    # include
    p_include = subparsers.add_parser("include", help="Undo shot exclusion")
    p_include.add_argument("episode", help="Episode stem")
    p_include.add_argument("shot", help="Shot number")

    # edit-caption
    p_edit = subparsers.add_parser("edit-caption", help="Edit a shot's caption")
    p_edit.add_argument("episode", help="Episode stem")
    p_edit.add_argument("shot", help="Shot number")
    p_edit.add_argument("caption", help="New caption text")

    # rebuild-dataset
    subparsers.add_parser("rebuild-dataset", help="Regenerate all dataset files")

    # summary
    subparsers.add_parser("summary", help="Print dataset statistics")

    # run
    p_run = subparsers.add_parser("run", help="Extract then summary")
    p_run.add_argument(
        "--episode", type=str, default=None, help="Single episode stem to process"
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "caption": cmd_caption,
        "extract": cmd_extract,
        "swap": cmd_swap,
        "exclude": cmd_exclude,
        "include": cmd_include,
        "edit-caption": cmd_edit_caption,
        "rebuild-dataset": cmd_rebuild_dataset,
        "summary": cmd_summary,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
