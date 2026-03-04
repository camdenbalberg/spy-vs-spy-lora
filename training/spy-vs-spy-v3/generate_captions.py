"""Generate AI training captions for Spy vs Spy episodes.

Phase A (grid mode): Extract unique frames, build comic grids, generate episode summaries.
Phase B (caption mode): Select sharpest frames, caption them using episode summaries as context.

Usage:
    python generate_captions.py grid --episode s1_e01_a     # Phase A: one episode
    python generate_captions.py grid --all                   # Phase A: all episodes
    python generate_captions.py grid --all --dry-run         # Grids only, no API calls
    python generate_captions.py caption --episode s1_e01_a   # Phase B: one episode
    python generate_captions.py caption --all                # Phase B: all
    python generate_captions.py caption --all --frames-per-episode 12
    python generate_captions.py status                       # Show progress
"""

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import anthropic
except ImportError:
    anthropic = None

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
EPISODES_DIR = SCRIPT_DIR / "raw" / "episodes"
SUMMARIES_DIR = SCRIPT_DIR / "raw" / "summaries"
GRIDS_DIR = SUMMARIES_DIR / "grids"
FRAMES_CACHE_DIR = SUMMARIES_DIR / "_frames"
CAPTIONS_DIR = SCRIPT_DIR / "raw" / "captions"
CAPTIONS_FRAMES_DIR = CAPTIONS_DIR / "frames"
CAPTION_PROMPT_PATH = SCRIPT_DIR / "gemini_spyvspy_caption_prompt.md"

# === Grid settings ===
GRID_COLS = 4
GRID_ROWS = 5
GRID_TARGET_FRAMES = GRID_COLS * GRID_ROWS  # 20
PANEL_W = 320
PANEL_H = 240

# === Frame extraction / dedup ===
DEDUP_MAD_THRESHOLD = 2.0  # Mean absolute difference below this = duplicate frame

# === Intro detection ===
# S1 title card: Black Spy on white (left) / White Spy on black (right).
# Detected via 16x16 perceptual hash comparison against this reference.
S1_TITLE_HASH = bytes.fromhex(
    "ff00ff00ff00e318e008ead8f138f118ff50e7d8f748e628e700e300e700c700"
)
S1_TITLE_HASH_THRESHOLD = 20  # max Hamming distance (title=0-8, content=100+)
# S2 title card: gray rectangle with "SPY vs SPY" text and both spies.
# Detected via 16x16 perceptual hash comparison against this reference.
S2_TITLE_HASH = bytes.fromhex(
    "ffffc003d001f203eb03ef03ee03cd03cc03de03d003c003c1cfc3ffffffffff"
)
S2_TITLE_HASH_THRESHOLD = 20  # max Hamming distance (title=0-7, content=150+)
# After title card, skip dark/fade frames until content begins
DARK_FRAME_THRESHOLD = 15.0    # mean brightness below this = dark/fade frame

# === Outro detection ===
# Trailing junk: DVD bumpers, MADtv logos, "Coming Soon" text, live-action bumpers.
# Detect by scanning backward — junk is dark OR colored (Spy vs Spy is B&W).
OUTRO_SATURATION_THRESHOLD = 50.0  # mean sat above this = colored (not B&W animation)

# === Scene detection ===
HIST_DIFF_THRESHOLD = 0.35
MIN_SCENE_GAP = 3  # minimum unique frames between scene changes

# === Sharpness / quality ===
LAPLACIAN_BLUR_THRESHOLD = 50
MAX_SATURATION_MEAN = 15.0  # frames with mean saturation above this have color bleed

# === Caption format (from frame_curator.py) ===
PREFIX = "spyvspy, "
SUFFIX_BOTH = (
    ", both wearing fedora hats and trenchcoats with long pointed beak noses"
    " and black sclera eyes, black and white ink comic art, bold outlines,"
    " high contrast, slapstick cartoon style"
)
SUFFIX_SINGLE = (
    ", wearing a fedora hat and trenchcoat with a long pointed beak nose"
    " and black sclera eyes, black and white ink comic art, bold outlines,"
    " high contrast, slapstick cartoon style"
)
SUFFIX_NONE = (
    ", black and white ink comic art, bold outlines, high contrast,"
    " slapstick cartoon style"
)

# === Caption frame selection ===
DEFAULT_FRAMES_PER_EPISODE = 10
MIN_FRAMES_PER_EPISODE = 6
MAX_FRAMES_PER_EPISODE = 15

# === API settings ===
API_MODEL = "claude-sonnet-4-20250514"
API_MAX_TOKENS_SUMMARY = 1000
API_MAX_TOKENS_CAPTION = 300
API_TEMPERATURE = 0.3
API_DELAY = 3

# === Episode summary prompt ===
SUMMARY_SYSTEM_PROMPT = """\
You are analyzing a comic grid of sequential frames from a Spy vs Spy animated episode.
The frames are arranged in reading order (left to right, top to bottom) and numbered 1 through N.

Spy vs Spy features two cartoon spies:
- Black spy: dressed in all black (hat, trenchcoat)
- White spy: dressed in all white (hat, trenchcoat)
- Occasionally: Grey spy (female, dressed in grey)

They have long pointed beak-like noses, black sclera eyes with white pupils, and engage \
in slapstick violence where each spy tries to outsmart and defeat the other.

Write a detailed LINEAR EPISODE SUMMARY describing what happens from start to finish. Include:
1. The setup (what is each spy doing at the start?)
2. The plan/trick (who sets up a trap or scheme?)
3. Key actions panel by panel
4. The climax (what goes wrong or right?)
5. The resolution (who wins? how do they celebrate?)

Reference panel numbers when describing events. Be specific about props, weapons, \
locations, and actions.

If existing captions from previously-analyzed frames are provided, use them as anchor \
points — they are accurate descriptions of specific moments in this episode."""


# ============================================================
# FFmpeg Utilities
# ============================================================

def get_duration(mkv_path: Path) -> float:
    """Get video duration via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mkv_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def get_fps(mkv_path: Path) -> float:
    """Get video FPS via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mkv_path)],
        capture_output=True, text=True,
    )
    # r_frame_rate is a fraction like "24000/1001"
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def extract_all_frames(mkv_path: Path, output_dir: Path) -> list[Path]:
    """Extract all frames at native FPS with deinterlacing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "raw_%06d.jpg")
    # yadif=0: output one frame per input frame (deinterlace without doubling)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mkv_path),
         "-vf", "yadif=0", "-q:v", "2", pattern],
        capture_output=True, text=True,
    )
    return sorted(output_dir.glob("raw_*.jpg"))


# ============================================================
# Frame Deduplication
# ============================================================

def deduplicate_frames(raw_frames: list[Path], output_dir: Path,
                       fps: float) -> tuple[list[Path], list[dict]]:
    """Compare consecutive frames and keep only unique ones.

    Returns (unique_frame_paths, metadata_list) where metadata has
    frame index, source timestamp, and sharpness score.
    """
    if not raw_frames:
        return [], []

    unique_paths = []
    metadata = []
    prev_small = None
    idx = 0

    for i, fp in enumerate(raw_frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Downscale for fast comparison
        small = cv2.resize(img, (160, 120))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_small is not None:
            mad = np.mean(np.abs(gray_small - prev_small))
            if mad < DEDUP_MAD_THRESHOLD:
                # Duplicate — skip
                continue

        prev_small = gray_small

        # This frame is unique — save it
        idx += 1
        out_path = output_dir / f"frame_{idx:05d}.jpg"
        shutil.copy2(str(fp), str(out_path))

        timestamp = i / fps if fps > 0 else 0.0
        sharpness = compute_sharpness(img)
        saturation = compute_saturation(img)

        unique_paths.append(out_path)
        metadata.append({
            "index": idx,
            "raw_frame": i + 1,
            "timestamp": round(timestamp, 3),
            "sharpness": round(sharpness, 1),
            "saturation": round(saturation, 1),
        })

    return unique_paths, metadata


# ============================================================
# Scene Detection
# ============================================================

def compute_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance as sharpness metric."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_saturation(frame: np.ndarray) -> float:
    """Mean saturation — high values indicate color bleed from live-action."""
    if len(frame.shape) < 3:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def is_good_quality(frame: np.ndarray) -> bool:
    """Check if a frame is usable: sharp enough and no color contamination."""
    sharpness = compute_sharpness(frame)
    saturation = compute_saturation(frame)
    return sharpness >= LAPLACIAN_BLUR_THRESHOLD and saturation <= MAX_SATURATION_MEAN


def _perceptual_hash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    """Compute a 16x16 average perceptual hash (256 bits)."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean_val = np.mean(small)
    return (small > mean_val).flatten().astype(np.uint8)


def _hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Count differing bits between two hash arrays."""
    return int(np.sum(hash1 != hash2))


# Unpack reference hashes once at module load
_S1_REF_HASH = np.unpackbits(np.frombuffer(S1_TITLE_HASH, dtype=np.uint8))
_S2_REF_HASH = np.unpackbits(np.frombuffer(S2_TITLE_HASH, dtype=np.uint8))


def is_title_card_s1(frame: np.ndarray) -> bool:
    """Detect S1 title card via perceptual hash comparison.

    The S1 title card shows Black Spy on white (left half) and White Spy
    on black (right half). Also catches fade-to-black variants because
    the perceptual hash pattern is preserved during the fade.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h = _perceptual_hash(gray)
    return _hamming_distance(h, _S1_REF_HASH) < S1_TITLE_HASH_THRESHOLD


def is_title_card_s2(frame: np.ndarray) -> bool:
    """Detect S2 title card via perceptual hash comparison.

    The S2 title card shows a gray rectangle with stylized "SPY vs SPY"
    text and both spies at the bottom. We compare against a reference
    hash — Hamming distance < 20 means it's the title card.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h = _perceptual_hash(gray)
    return _hamming_distance(h, _S2_REF_HASH) < S2_TITLE_HASH_THRESHOLD


def detect_intro_end(frame_paths: list[Path]) -> int:
    """Find where episode content begins by locating the title card.

    Tries both S1 detection (perceptual hash) and S2 detection
    (perceptual hash). Scans forward, finds the LAST title card frame,
    then skips any dark/fade-to-black frames that follow.

    Returns 0 if no title card detected (assumes no intro to skip).
    """
    last_title_idx = -1
    # Intro is never more than 25% of total frames
    scan_limit = min(len(frame_paths), max(150, len(frame_paths) // 4))

    for i in range(scan_limit):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            continue
        if is_title_card_s1(frame) or is_title_card_s2(frame):
            last_title_idx = i

    if last_title_idx < 0:
        return 0

    # After the last title card, skip dark/fade-to-black frames.
    # The title card often fades to black before content begins.
    content_start = last_title_idx + 1
    while content_start < len(frame_paths):
        frame = cv2.imread(str(frame_paths[content_start]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            content_start += 1
            continue
        mean_brightness = float(np.mean(frame))
        if mean_brightness > DARK_FRAME_THRESHOLD:
            break
        content_start += 1

    return content_start


def detect_outro_start(frame_paths: list[Path]) -> int:
    """Find where episode content ends and outro junk begins.

    Scans backward from the end. A trailing frame is junk if it's either
    too dark (fade/black) or has high color saturation (DVD bumpers,
    MADtv logos, live-action footage — Spy vs Spy is B&W).

    Returns the index of the first junk frame, or len(frame_paths) if
    no junk detected.
    """
    n = len(frame_paths)
    if n == 0:
        return n

    junk_start = n
    for i in range(n - 1, max(0, n - 250) - 1, -1):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            junk_start = i
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = float(np.mean(gray))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))

        if bright < DARK_FRAME_THRESHOLD or sat > OUTRO_SATURATION_THRESHOLD:
            junk_start = i
        else:
            break

    return junk_start


def detect_scene_changes(frame_paths: list[Path]) -> list[int]:
    """Find indices in frame_paths where scene changes occur."""
    if len(frame_paths) < 2:
        return list(range(len(frame_paths)))

    diffs = []
    prev_hist = None

    for fp in frame_paths:
        gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            diffs.append(0.0)
            continue
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            diffs.append(float(diff))
        else:
            diffs.append(0.0)

        prev_hist = hist

    # Find peaks above threshold with minimum gap
    peaks = []
    for i, d in enumerate(diffs):
        if d >= HIST_DIFF_THRESHOLD:
            if not peaks or (i - peaks[-1]) >= MIN_SCENE_GAP:
                peaks.append(i)

    return peaks


# ============================================================
# Grid Generation
# ============================================================

def _is_frame_usable(meta: dict) -> bool:
    """Check metadata for quality: sharp enough, no color bleed."""
    return (meta["sharpness"] >= LAPLACIAN_BLUR_THRESHOLD
            and meta.get("saturation", 0) <= MAX_SATURATION_MEAN)


def select_grid_frames(scene_indices: list[int], all_frames: list[Path],
                       metadata: list[dict], intro_end: int = 0,
                       target_count: int = GRID_TARGET_FRAMES) -> list[int]:
    """Select target_count frame indices for the comic grid.

    Prefers scene-change frames that are sharp and within the usable range
    (after intro, before outro fade). Supplements with evenly-spaced frames
    if needed.
    """
    start = intro_end
    end = len(all_frames)

    # Score scene-change frames by quality
    candidates = []
    for idx in scene_indices:
        if start <= idx < end and idx < len(metadata):
            if _is_frame_usable(metadata[idx]):
                candidates.append(idx)

    # Supplement with evenly-spaced usable frames if not enough
    if len(candidates) < target_count:
        existing = set(candidates)
        step = max(1, (end - start) // (target_count * 2))
        for idx in range(start, end, step):
            if idx not in existing and idx < len(metadata):
                if _is_frame_usable(metadata[idx]):
                    candidates.append(idx)
                    existing.add(idx)

    # Sort chronologically
    candidates.sort()

    # If still more than target, select evenly distributed
    if len(candidates) > target_count:
        indices = np.linspace(0, len(candidates) - 1, target_count, dtype=int)
        candidates = [candidates[i] for i in indices]

    # Clamp to target
    return candidates[:target_count]


def build_grid_image(frame_paths: list[Path], selected_indices: list[int],
                     episode_name: str) -> Image.Image:
    """Arrange selected frames into a numbered comic grid."""
    n_panels = len(selected_indices)
    cols = GRID_COLS
    rows = math.ceil(n_panels / cols)

    gap = 2
    label_h = 30
    total_w = cols * PANEL_W + (cols - 1) * gap
    total_h = label_h + rows * PANEL_H + (rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a monospace font for labels
    try:
        font = ImageFont.truetype("consola.ttf", 18)
        small_font = ImageFont.truetype("consola.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font

    draw.text((10, 5), episode_name, fill=(0, 0, 0), font=font)

    for panel_num, frame_idx in enumerate(selected_indices):
        col = panel_num % cols
        row = panel_num // cols
        x = col * (PANEL_W + gap)
        y = label_h + row * (PANEL_H + gap)

        fp = frame_paths[frame_idx]
        panel = Image.open(fp).convert("RGB")
        panel = panel.resize((PANEL_W, PANEL_H), Image.LANCZOS)
        canvas.paste(panel, (x, y))

        # Panel number with dark outline for visibility on B&W art
        num_text = str(panel_num + 1)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + 4 + dx, y + 2 + dy), num_text,
                              fill=(0, 0, 0), font=small_font)
        draw.text((x + 4, y + 2), num_text, fill=(255, 255, 0), font=small_font)

    return canvas



# ============================================================
# Caption Wrapping (from frame_curator.py)
# ============================================================

def wrap_caption(response_text: str) -> str:
    """Wrap raw caption with spyvspy prefix and style suffix."""
    t = response_text.lower()
    has_black = "black spy" in t
    has_white = "white spy" in t
    has_grey = "grey spy" in t or "gray spy" in t
    spy_count = sum([has_black, has_white, has_grey])
    if spy_count >= 2:
        suffix = SUFFIX_BOTH
    elif spy_count == 1:
        suffix = SUFFIX_SINGLE
    else:
        suffix = SUFFIX_NONE
    return PREFIX + response_text.strip().rstrip(".") + suffix


# ============================================================
# Claude API Client
# ============================================================

def get_api_key() -> str:
    """Resolve API key from environment or Windows registry."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as key:
                api_key, _ = winreg.QueryValueEx(key, "ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        print("ANTHROPIC_API_KEY not found. Set it as a system environment variable.")
        sys.exit(1)
    return api_key


def call_claude_vision(client, system_prompt: str, image_path: Path,
                       user_text: str, max_tokens: int,
                       max_retries: int = 5) -> str:
    """Send image + text to Claude Vision API with retry logic."""
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lower()
    media = "image/png" if ext == ".png" else "image/jpeg"

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=API_MODEL,
                max_tokens=max_tokens,
                temperature=API_TEMPERATURE,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": media, "data": data}},
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise Exception(f"Failed after {max_retries} retries")


# ============================================================
# Frame Extraction + Caching
# ============================================================

def extract_and_cache_unique_frames(segment_name: str,
                                    mkv_path: Path) -> tuple[list[Path], list[dict], int]:
    """Extract all unique frames for an episode, caching results.

    Returns (unique_frame_paths, frame_metadata_list, intro_end_index).
    Cached in FRAMES_CACHE_DIR/{segment_name}/.
    """
    cache_dir = FRAMES_CACHE_DIR / segment_name
    meta_path = cache_dir / "metadata.json"

    # Check cache
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            saved = json.load(f)
        # Support both old (list) and new (dict) format
        if isinstance(saved, dict):
            metadata = saved["frames"]
            intro_end = saved.get("intro_end", 0)
        else:
            metadata = saved
            intro_end = 0
        unique_paths = sorted(cache_dir.glob("frame_*.jpg"))
        if len(unique_paths) == len(metadata):
            print(f"    Loaded {len(unique_paths)} cached unique frames "
                  f"(intro ends at frame {intro_end})")
            return unique_paths, metadata, intro_end

    # Extract all frames at native FPS
    raw_dir = cache_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(mkv_path)
    duration = get_duration(mkv_path)
    expected_frames = int(fps * duration)
    print(f"    Extracting all frames ({fps:.2f} FPS, ~{expected_frames} expected)...")

    raw_frames = extract_all_frames(mkv_path, raw_dir)
    print(f"    Extracted {len(raw_frames)} raw frames")

    # Deduplicate
    print(f"    Deduplicating (MAD threshold={DEDUP_MAD_THRESHOLD})...")
    unique_paths, metadata = deduplicate_frames(raw_frames, cache_dir, fps)
    print(f"    {len(unique_paths)} unique frames "
          f"({len(raw_frames) - len(unique_paths)} duplicates removed)")

    # Detect intro end (find SPY vs SPY title card)
    print(f"    Detecting intro sequence...")
    intro_end = detect_intro_end(unique_paths)
    if intro_end > 0:
        intro_ts = metadata[intro_end - 1]["timestamp"] if intro_end <= len(metadata) else 0
        print(f"    Intro ends at frame {intro_end} "
              f"({intro_ts:.1f}s, title card at frame {intro_end - 1})")
    else:
        print(f"    No intro detected")

    # Save metadata (new dict format with intro_end)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"intro_end": intro_end, "frames": metadata}, f, indent=2)

    # Clean up raw frames
    shutil.rmtree(raw_dir, ignore_errors=True)

    return unique_paths, metadata, intro_end


# ============================================================
# Phase A: Grid + Summary
# ============================================================

def phase_a_process_episode(client, segment_name: str, mkv_path: Path,
                            dry_run: bool = False,
                            delay: float = API_DELAY) -> dict:
    """Full Phase A pipeline for one episode."""
    print(f"\n  Phase A: {segment_name}")

    # Check if summary already exists (resume support)
    summary_path = SUMMARIES_DIR / f"{segment_name}_summary.txt"
    grid_path = GRIDS_DIR / f"{segment_name}_grid.jpg"
    if summary_path.exists() and not dry_run:
        print(f"    Summary already exists, skipping")
        return {"status": "skipped", "summary_path": str(summary_path)}

    duration = get_duration(mkv_path)
    print(f"    Duration: {duration:.1f}s")

    # Step 1: Extract unique frames
    unique_paths, metadata, intro_end = extract_and_cache_unique_frames(
        segment_name, mkv_path)

    if not unique_paths:
        print(f"    ERROR: No frames extracted")
        return {"status": "error", "reason": "no_frames"}

    # Step 2: Scene change detection (only on content frames, after intro)
    content_paths = unique_paths[intro_end:]
    scene_indices_raw = detect_scene_changes(content_paths)
    # Offset indices back to full-frame numbering
    scene_indices = [idx + intro_end for idx in scene_indices_raw]
    print(f"    Detected {len(scene_indices)} scene changes (after intro)")

    # Step 3: Select grid frames
    selected = select_grid_frames(scene_indices, unique_paths, metadata,
                                  intro_end=intro_end)
    print(f"    Selected {len(selected)} frames for grid")

    # Step 4: Build grid image
    grid_image = build_grid_image(unique_paths, selected, segment_name)
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    grid_image.save(str(grid_path), "JPEG", quality=92)
    print(f"    Grid saved: {grid_path.name}")

    if dry_run:
        return {
            "status": "dry_run",
            "grid_path": str(grid_path),
            "panels": len(selected),
            "unique_frames": len(unique_paths),
        }

    # Step 5: Build user message and send to Claude Vision
    n_panels = len(selected)
    user_text = (
        f"This is a {n_panels}-panel comic grid from Spy vs Spy episode "
        f"'{segment_name}' ({duration:.0f}s).\n"
        f"Panels are numbered 1-{n_panels}, reading left to right, top to bottom.\n\n"
        "Describe the complete narrative from what you see.\n"
    )

    summary_text = call_claude_vision(
        client, SUMMARY_SYSTEM_PROMPT, grid_path, user_text, API_MAX_TOKENS_SUMMARY,
    )

    # Step 7: Save summary
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"    Summary saved: {summary_path.name} ({len(summary_text)} chars)")

    time.sleep(delay)

    return {
        "status": "ok",
        "grid_path": str(grid_path),
        "summary_path": str(summary_path),
        "panels": n_panels,
        "unique_frames": len(unique_paths),
        "summary_length": len(summary_text),
    }


# ============================================================
# Phase B: Sharp Frame Selection + Captioning
# ============================================================

def select_sharp_diverse_frames(metadata: list[dict],
                                n_frames: int,
                                intro_end: int = 0) -> list[int]:
    """Select top-N sharpest + temporally diverse frame indices.

    Starts after the intro, filters out color-contaminated and blurry
    frames, then divides usable range into n_frames temporal bins and
    picks the sharpest frame from each bin.
    """
    if not metadata:
        return []

    start = intro_end
    end = len(metadata)
    usable = metadata[start:end]

    if not usable:
        return []

    n_bins = min(n_frames, len(usable))
    bin_size = len(usable) / n_bins
    selected = []

    for b in range(n_bins):
        bin_start = int(b * bin_size)
        bin_end = int((b + 1) * bin_size)
        bin_frames = usable[bin_start:bin_end]
        if not bin_frames:
            continue
        # Pick the sharpest usable frame in this bin
        usable_in_bin = [(i, m) for i, m in enumerate(bin_frames)
                         if _is_frame_usable(m)]
        if not usable_in_bin:
            continue
        best_i, _ = max(usable_in_bin, key=lambda x: x[1]["sharpness"])
        selected.append(start + bin_start + best_i)

    return selected


def phase_b_process_episode(client, segment_name: str, mkv_path: Path,
                            caption_prompt: str,
                            n_frames: int, dry_run: bool = False,
                            delay: float = API_DELAY) -> dict:
    """Full Phase B pipeline for one episode."""
    print(f"\n  Phase B: {segment_name}")

    # Load summary from Phase A (required)
    summary_path = SUMMARIES_DIR / f"{segment_name}_summary.txt"
    if not summary_path.exists():
        print(f"    ERROR: No summary found. Run 'grid' mode first.")
        return {"status": "error", "reason": "no_summary"}

    summary_text = summary_path.read_text(encoding="utf-8")

    # Check existing captions (resume support)
    existing_captions = list(CAPTIONS_DIR.glob(f"{segment_name}_*.txt"))
    if existing_captions and not dry_run:
        print(f"    {len(existing_captions)} captions already exist, skipping")
        return {"status": "skipped", "existing": len(existing_captions)}

    # Step 1: Load cached unique frames from Phase A
    cache_dir = FRAMES_CACHE_DIR / segment_name
    meta_path = cache_dir / "metadata.json"

    if not meta_path.exists():
        # Need to extract first
        unique_paths, metadata, intro_end = extract_and_cache_unique_frames(
            segment_name, mkv_path)
    else:
        with open(meta_path, encoding="utf-8") as f:
            saved = json.load(f)
        # Support both old (list) and new (dict) format
        if isinstance(saved, dict):
            metadata = saved["frames"]
            intro_end = saved.get("intro_end", 0)
        else:
            metadata = saved
            intro_end = 0
        unique_paths = sorted(cache_dir.glob("frame_*.jpg"))
        print(f"    Loaded {len(unique_paths)} cached unique frames "
              f"(intro ends at frame {intro_end})")

    if not unique_paths:
        print(f"    ERROR: No frames available")
        return {"status": "error", "reason": "no_frames"}

    duration = get_duration(mkv_path)

    # Step 2: Select sharp + diverse frames
    selected_indices = select_sharp_diverse_frames(metadata, n_frames,
                                                   intro_end=intro_end)
    print(f"    Selected {len(selected_indices)} sharp frames "
          f"from {len(unique_paths)} unique ({duration:.1f}s episode)")

    if dry_run:
        for i, idx in enumerate(selected_indices):
            m = metadata[idx]
            print(f"      [{i + 1}] frame_{m['index']:05d} "
                  f"@ {m['timestamp']:.1f}s  sharpness={m['sharpness']:.0f}")
        return {"status": "dry_run", "frames": len(selected_indices)}

    # Step 3: Caption each selected frame
    CAPTIONS_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, idx in enumerate(selected_indices):
        m = metadata[idx]
        frame_name = f"{segment_name}_{i + 1:03d}"
        src_path = unique_paths[idx]
        out_frame = CAPTIONS_FRAMES_DIR / f"{frame_name}.jpg"

        # Copy frame to output
        shutil.copy2(str(src_path), str(out_frame))

        # Build context-rich user message
        user_text = (
            f"EPISODE CONTEXT: This frame is from '{segment_name}', "
            f"approximately {m['timestamp']:.1f}s into a {duration:.0f}s episode.\n\n"
            f"EPISODE SUMMARY:\n{summary_text}\n\n"
            f"Using the episode summary as narrative context, caption this specific "
            f"frame. Focus on what is visible in THIS frame only. Do not describe "
            f"events from other parts of the episode."
        )

        try:
            raw_caption = call_claude_vision(
                client, caption_prompt, out_frame, user_text, API_MAX_TOKENS_CAPTION,
            )
            wrapped = wrap_caption(raw_caption)

            caption_path = CAPTIONS_DIR / f"{frame_name}.txt"
            caption_path.write_text(wrapped, encoding="utf-8")

            results.append({
                "frame": frame_name,
                "timestamp": m["timestamp"],
                "sharpness": m["sharpness"],
                "caption_length": len(wrapped),
                "status": "ok",
            })
            print(f"    [{i + 1}/{len(selected_indices)}] {frame_name} "
                  f"@ {m['timestamp']:.1f}s (sharpness={m['sharpness']:.0f})")

        except Exception as e:
            print(f"    [{i + 1}/{len(selected_indices)}] {frame_name} ERROR: {e}")
            results.append({
                "frame": frame_name, "status": "error", "error": str(e),
            })

        time.sleep(delay)

    return {
        "status": "ok",
        "frames_captioned": sum(1 for r in results if r["status"] == "ok"),
        "frames_failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


# ============================================================
# Progress Tracking
# ============================================================

def save_progress(mode: str, results: dict):
    """Save progress to JSON manifest."""
    if mode == "grid":
        manifest_path = SUMMARIES_DIR / "summaries.json"
    else:
        manifest_path = CAPTIONS_DIR / "captions_manifest.json"

    existing = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)

    existing.update(results)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def show_status():
    """Show progress summary for both phases."""
    episodes = sorted(EPISODES_DIR.glob("*.mkv"))
    segment_names = [e.stem for e in episodes]

    # Phase A
    summaries = set()
    if SUMMARIES_DIR.exists():
        for f in SUMMARIES_DIR.glob("*_summary.txt"):
            summaries.add(f.stem.replace("_summary", ""))
    grids = set()
    if GRIDS_DIR.exists():
        for f in GRIDS_DIR.glob("*_grid.jpg"):
            grids.add(f.stem.replace("_grid", ""))

    # Cached frames
    cached = set()
    if FRAMES_CACHE_DIR.exists():
        for d in FRAMES_CACHE_DIR.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                cached.add(d.name)

    # Phase B
    captioned_episodes = set()
    total_captions = 0
    if CAPTIONS_DIR.exists():
        for f in CAPTIONS_DIR.glob("*.txt"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                captioned_episodes.add(parts[0])
                total_captions += 1

    print(f"Episodes: {len(segment_names)}")
    print(f"Frames cached:       {len(cached)}/{len(segment_names)}")
    print(f"Phase A - Grids:     {len(grids)}/{len(segment_names)}")
    print(f"Phase A - Summaries: {len(summaries)}/{len(segment_names)}")
    print(f"Phase B - Captioned: {len(captioned_episodes)}/{len(segment_names)} episodes, {total_captions} frames")

    missing_summaries = sorted(s for s in segment_names if s not in summaries)
    missing_captions = sorted(s for s in segment_names if s not in captioned_episodes)

    if missing_summaries:
        shown = missing_summaries[:10]
        extra = len(missing_summaries) - len(shown)
        print(f"\nMissing summaries: {', '.join(shown)}"
              + (f" (+{extra} more)" if extra else ""))
    if missing_captions:
        shown = missing_captions[:10]
        extra = len(missing_captions) - len(shown)
        print(f"Missing captions:  {', '.join(shown)}"
              + (f" (+{extra} more)" if extra else ""))


# ============================================================
# Intro Cleanup
# ============================================================

def clean_intros_for_episode(segment_name: str) -> dict:
    """Re-detect intro and delete intro frames from cached folder.

    Only deletes frames confirmed as intro (before the title card).
    Episodes with no intro detected are left completely untouched.
    """
    cache_dir = FRAMES_CACHE_DIR / segment_name
    meta_path = cache_dir / "metadata.json"

    if not meta_path.exists():
        return {"status": "no_cache"}

    with open(meta_path, encoding="utf-8") as f:
        saved = json.load(f)

    if isinstance(saved, dict):
        metadata = saved["frames"]
        old_intro_end = saved.get("intro_end", 0)
    else:
        metadata = saved
        old_intro_end = 0

    unique_paths = sorted(cache_dir.glob("frame_*.jpg"))
    if len(unique_paths) != len(metadata):
        return {"status": "mismatch",
                "files": len(unique_paths), "meta": len(metadata)}

    # Re-detect intro with both S1 and S2 detection
    intro_end = detect_intro_end(unique_paths)

    if intro_end == 0:
        # No intro found — do NOT touch any frames
        if old_intro_end != 0:
            # Update metadata to reflect no intro
            saved_out = {"intro_end": 0, "frames": metadata}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(saved_out, f, indent=2)
        return {"status": "no_intro", "frames": len(unique_paths)}

    # Delete intro frame files
    deleted = 0
    for i in range(intro_end):
        fp = unique_paths[i]
        if fp.exists():
            fp.unlink()
            deleted += 1

    # Update metadata: remove intro entries, set intro_end to 0
    content_metadata = metadata[intro_end:]
    saved_out = {"intro_end": 0, "frames": content_metadata}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(saved_out, f, indent=2)

    return {
        "status": "cleaned",
        "intro_frames_deleted": deleted,
        "content_frames": len(content_metadata),
        "intro_end_was": intro_end,
    }


# ============================================================
# CLI / Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate AI training captions for Spy vs Spy episodes",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Grid mode (Phase A)
    grid_p = subparsers.add_parser(
        "grid", help="Phase A: Generate comic grids + episode summaries")
    grid_p.add_argument("--episode", help="Process single episode (e.g. s1_e01_a)")
    grid_p.add_argument("--all", action="store_true", help="Process all episodes")
    grid_p.add_argument("--dry-run", action="store_true",
                        help="Generate grids only, no API calls")
    grid_p.add_argument("--delay", type=float, default=API_DELAY,
                        help=f"Delay between API calls in seconds (default: {API_DELAY})")

    # Caption mode (Phase B)
    cap_p = subparsers.add_parser(
        "caption", help="Phase B: Select sharp frames + generate captions")
    cap_p.add_argument("--episode", help="Process single episode (e.g. s1_e01_a)")
    cap_p.add_argument("--all", action="store_true", help="Process all episodes")
    cap_p.add_argument("--dry-run", action="store_true",
                       help="Select frames only, no API calls")
    cap_p.add_argument("--frames-per-episode", type=int, default=None,
                       help="Override frames per episode (default: auto by duration)")
    cap_p.add_argument("--delay", type=float, default=API_DELAY,
                       help=f"Delay between API calls in seconds (default: {API_DELAY})")

    # Status mode
    subparsers.add_parser("status", help="Show progress summary")

    # Clean intros mode
    clean_p = subparsers.add_parser(
        "clean-intros",
        help="Re-detect and delete intro frames from cached episode folders")
    clean_p.add_argument("--dry-run", action="store_true",
                         help="Show what would be deleted without deleting")

    # Clean outros mode
    clean_o = subparsers.add_parser(
        "clean-outros",
        help="Detect and delete trailing junk (bumpers, logos, live-action)")
    clean_o.add_argument("--dry-run", action="store_true",
                         help="Show what would be deleted without deleting")

    args = parser.parse_args()

    if args.mode == "status":
        show_status()
        return

    if args.mode == "clean-intros":
        # Process all cached episode folders
        if not FRAMES_CACHE_DIR.exists():
            print("No cached frames found.")
            return
        cached_dirs = sorted(
            d for d in FRAMES_CACHE_DIR.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
        print(f"Scanning {len(cached_dirs)} cached episodes for intros"
              + (" (dry run)" if args.dry_run else ""))
        total_deleted = 0
        total_cleaned = 0
        total_no_intro = 0
        for d in cached_dirs:
            seg_name = d.name
            if args.dry_run:
                # Just detect, don't delete
                with open(d / "metadata.json", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    metadata = saved["frames"]
                else:
                    metadata = saved
                paths = sorted(d.glob("frame_*.jpg"))
                if len(paths) != len(metadata):
                    print(f"  {seg_name}: MISMATCH files={len(paths)} meta={len(metadata)}")
                    continue
                intro_end = detect_intro_end(paths)
                if intro_end > 0:
                    ts = metadata[intro_end - 1]["timestamp"]
                    print(f"  {seg_name}: would delete {intro_end} intro frames "
                          f"(title card at {ts:.1f}s), keeping {len(paths) - intro_end}")
                    total_deleted += intro_end
                    total_cleaned += 1
                else:
                    print(f"  {seg_name}: no intro (keeping all {len(paths)} frames)")
                    total_no_intro += 1
            else:
                result = clean_intros_for_episode(seg_name)
                if result["status"] == "cleaned":
                    print(f"  {seg_name}: deleted {result['intro_frames_deleted']} intro frames, "
                          f"{result['content_frames']} content frames remain")
                    total_deleted += result["intro_frames_deleted"]
                    total_cleaned += 1
                elif result["status"] == "no_intro":
                    print(f"  {seg_name}: no intro detected, untouched "
                          f"({result['frames']} frames)")
                    total_no_intro += 1
                else:
                    print(f"  {seg_name}: {result['status']}")
        print(f"\nDone: {total_cleaned} episodes cleaned, "
              f"{total_deleted} intro frames {'would be ' if args.dry_run else ''}deleted, "
              f"{total_no_intro} episodes had no intro")
        return

    if args.mode == "clean-outros":
        if not FRAMES_CACHE_DIR.exists():
            print("No cached frames found.")
            return
        cached_dirs = sorted(
            d for d in FRAMES_CACHE_DIR.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
        print(f"Scanning {len(cached_dirs)} cached episodes for outro junk"
              + (" (dry run)" if args.dry_run else ""))
        total_deleted = 0
        total_cleaned = 0
        total_clean = 0
        for d in cached_dirs:
            seg_name = d.name
            with open(d / "metadata.json", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                metadata = saved["frames"]
            else:
                metadata = saved
            paths = sorted(d.glob("frame_*.jpg"))
            if len(paths) != len(metadata):
                print(f"  {seg_name}: MISMATCH files={len(paths)} meta={len(metadata)}")
                continue

            outro_start = detect_outro_start(paths)
            junk_count = len(paths) - outro_start

            if junk_count == 0:
                print(f"  {seg_name}: clean ({len(paths)} frames)")
                total_clean += 1
                continue

            if args.dry_run:
                print(f"  {seg_name}: would delete {junk_count} outro frames, "
                      f"keeping {outro_start}")
                total_deleted += junk_count
                total_cleaned += 1
            else:
                # Delete outro frame files
                deleted = 0
                for i in range(outro_start, len(paths)):
                    if paths[i].exists():
                        paths[i].unlink()
                        deleted += 1
                # Update metadata
                content_metadata = metadata[:outro_start]
                saved_out = {"intro_end": 0, "frames": content_metadata}
                with open(d / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(saved_out, f, indent=2)
                print(f"  {seg_name}: deleted {deleted} outro frames, "
                      f"{outro_start} content frames remain")
                total_deleted += deleted
                total_cleaned += 1

        print(f"\nDone: {total_cleaned} episodes cleaned, "
              f"{total_deleted} outro frames {'would be ' if args.dry_run else ''}deleted, "
              f"{total_clean} episodes were already clean")
        return

    # Resolve episodes to process
    if hasattr(args, "episode") and args.episode:
        mkv_path = EPISODES_DIR / f"{args.episode}.mkv"
        if not mkv_path.exists():
            print(f"ERROR: {mkv_path} not found")
            sys.exit(1)
        episodes = [(args.episode, mkv_path)]
    elif hasattr(args, "all") and args.all:
        episodes = [(p.stem, p) for p in sorted(EPISODES_DIR.glob("*.mkv"))]
    else:
        print("Specify --episode or --all")
        sys.exit(1)

    print(f"Processing {len(episodes)} episode(s) in '{args.mode}' mode"
          + (" (dry run)" if args.dry_run else ""))

    dry_run = args.dry_run

    if not dry_run:
        if anthropic is None:
            print("ERROR: anthropic package not installed. pip install anthropic")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=get_api_key())
    else:
        client = None

    delay = args.delay

    if args.mode == "grid":
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        GRIDS_DIR.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for seg_name, mkv_path in episodes:
            result = phase_a_process_episode(
                client, seg_name, mkv_path, dry_run, delay)
            all_results[seg_name] = result
            save_progress("grid", {seg_name: result})

        ok = sum(1 for r in all_results.values() if r["status"] == "ok")
        skip = sum(1 for r in all_results.values() if r["status"] == "skipped")
        dry = sum(1 for r in all_results.values() if r["status"] == "dry_run")
        print(f"\nDone: {ok} summaries generated, {skip} skipped, {dry} dry-run")

    elif args.mode == "caption":
        CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
        CAPTIONS_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        caption_prompt = CAPTION_PROMPT_PATH.read_text(encoding="utf-8")

        all_results = {}
        for seg_name, mkv_path in episodes:
            duration = get_duration(mkv_path)
            if args.frames_per_episode:
                n = args.frames_per_episode
            else:
                n = max(MIN_FRAMES_PER_EPISODE,
                        min(MAX_FRAMES_PER_EPISODE, int(duration / 8)))

            result = phase_b_process_episode(
                client, seg_name, mkv_path, caption_prompt,
                n, dry_run, delay)
            all_results[seg_name] = result
            save_progress("caption", {seg_name: result})

        ok = sum(1 for r in all_results.values() if r["status"] == "ok")
        total_frames = sum(
            r.get("frames_captioned", 0) for r in all_results.values())
        print(f"\nDone: {ok} episodes captioned, {total_frames} total new frames")


if __name__ == "__main__":
    main()
