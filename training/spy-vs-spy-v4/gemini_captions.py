#!/usr/bin/env python3
"""Multi-pass Gemini video captioning for Spy vs Spy LoRA training.

Uploads each episode video to Gemini API, runs 5 focused analysis passes,
then merges results via Claude API into final LoRA training captions.

Usage:
    pip install google-genai anthropic
    export GEMINI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"

    python gemini_captions.py test s1_e01_a     # Test one episode, print results
    python gemini_captions.py run               # Process all episodes
    python gemini_captions.py run s1_e01_a      # Process specific episode
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google import genai
import anthropic


def _get_env(name):
    """Get environment variable, falling back to Windows system env if needed."""
    val = os.environ.get(name)
    if val:
        return val
    # Windows: shell may not have inherited system env vars
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell.exe", "-Command",
                 f"[System.Environment]::GetEnvironmentVariable('{name}', 'User')"],
                capture_output=True, text=True, timeout=5,
            )
            val = result.stdout.strip()
            if val:
                return val
            result = subprocess.run(
                ["powershell.exe", "-Command",
                 f"[System.Environment]::GetEnvironmentVariable('{name}', 'Machine')"],
                capture_output=True, text=True, timeout=5,
            )
            val = result.stdout.strip()
            if val:
                return val
        except Exception:
            pass
    return None

# ── Paths ────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
EPISODES_DIR = PROJECT_ROOT / "raw" / "gemini"
FRAMES_DIR = PROJECT_ROOT / "frames"
GEMINI_OUTPUT_DIR = PROJECT_ROOT / "captions" / "gemini_passes"

# ── Config ───────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-4-0"

PASS_FILES = [
    "pass1_scene_layout.txt",
    "pass2_expressions.txt",
    "pass3_outfits.txt",
    "pass4_objects.txt",
    "pass5_background.txt",
]

PASS_NAMES = [
    "Scene & Layout",
    "Expressions",
    "Outfits",
    "Objects",
    "Background",
]


# ── Prompt Loading ───────────────────────────────────────────────────────────────

def load_prompt(filename):
    """Load a prompt file from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        print(f"ERROR: Prompt file not found: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def load_merge_prompt():
    """Load the Claude merge prompt."""
    return load_prompt("claude_merge.txt")


def build_followup_prompt(pass_prompt, pass1_csv):
    """Build a follow-up pass prompt that includes pass 1 timestamps as anchors.

    Passes 2-5 receive the pass 1 output so they use the same scene timestamps.
    """
    anchor = (
        "The following scenes were identified in this video by a previous analysis. "
        "Use these EXACT timestamps — do not add or remove scenes:\n\n"
        f"{pass1_csv}\n\n"
        "Now, for each of those timestamps, provide your analysis:\n\n"
    )
    return anchor + pass_prompt


# ── Gemini API ───────────────────────────────────────────────────────────────────

def upload_video(client, video_path):
    """Upload a video file to Gemini Files API and wait until ready."""
    print(f"  Uploading {video_path.name}...")
    uploaded = client.files.upload(file=str(video_path))
    print(f"  Upload complete: {uploaded.name}")

    # Poll until file is processed (state == ACTIVE)
    while True:
        file_info = client.files.get(name=uploaded.name)
        state = str(file_info.state)
        if "ACTIVE" in state:
            break
        elif "FAILED" in state:
            print(f"  ERROR: File processing failed for {video_path.name}")
            sys.exit(1)
        print(f"  Waiting for processing (state: {state})...")
        time.sleep(2)

    print(f"  File ready.")
    return uploaded


def run_gemini_pass(client, video_file, prompt, pass_name):
    """Run a single Gemini pass on the uploaded video."""
    print(f"  Running {pass_name}...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[video_file, prompt],
    )
    text = response.text.strip()
    print(f"  {pass_name} complete ({len(text)} chars)")
    return text


def count_pass1_scenes(pass1_result):
    """Count how many CSV data rows are in a pass1 result."""
    count = 0
    for line in pass1_result.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("timestamp,"):
            continue
        # Should start with a timestamp like 0:00 or 0:05
        if len(line) > 3 and ":" in line[:5]:
            count += 1
    return count


PASS1_MIN_SCENES = 6  # Retry if pass1 finds fewer than this


def run_all_passes(client, video_path):
    """Upload video and run all 5 Gemini passes. Returns list of response strings."""
    video_file = upload_video(client, video_path)

    results = []

    # Pass 1 — Scene & Layout (establishes timestamps)
    pass1_prompt = load_prompt(PASS_FILES[0])
    pass1_result = run_gemini_pass(client, video_file, pass1_prompt, PASS_NAMES[0])

    scene_count = count_pass1_scenes(pass1_result)
    if scene_count < PASS1_MIN_SCENES:
        print(f"  WARNING: Pass 1 found only {scene_count} scenes (min {PASS1_MIN_SCENES}). Retrying...")
        retry_suffix = (
            f"\n\nIMPORTANT: Your previous attempt found only {scene_count} scenes. "
            f"This is far too few for this clip. Most clips have 10-25 camera cuts. "
            f"Please watch the video again very carefully and identify EVERY camera cut, "
            f"even brief 1-second shots. Do not combine multiple shots into one row."
        )
        pass1_result = run_gemini_pass(
            client, video_file, pass1_prompt + retry_suffix, f"{PASS_NAMES[0]} (retry)"
        )
        new_count = count_pass1_scenes(pass1_result)
        print(f"  Retry found {new_count} scenes (was {scene_count})")

    results.append(pass1_result)

    # Passes 2-5 — anchored to pass 1 timestamps
    for i in range(1, 5):
        pass_prompt = load_prompt(PASS_FILES[i])
        full_prompt = build_followup_prompt(pass_prompt, pass1_result)
        result = run_gemini_pass(client, video_file, full_prompt, PASS_NAMES[i])
        results.append(result)

    # Clean up uploaded file
    try:
        client.files.delete(name=video_file.name)
        print(f"  Cleaned up uploaded file.")
    except Exception:
        pass  # Not critical

    return results


# ── Claude Merge ─────────────────────────────────────────────────────────────────

def merge_with_claude(claude_client, pass_results, episode_name):
    """Send all 5 pass results to Claude for merging into final captions."""
    print(f"  Merging with Claude API...")

    merge_prompt = load_merge_prompt()

    # Build the user message with all 5 pass results
    user_message = f"Episode: {episode_name}\n\n"
    for i, (name, result) in enumerate(zip(PASS_NAMES, pass_results)):
        user_message += f"=== PASS {i+1}: {name} ===\n{result}\n\n"

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=merge_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()
    print(f"  Merge complete ({len(text)} chars)")
    return text


def parse_merged_output(merged_text):
    """Parse JSONL output from Claude into list of scene dicts."""
    scenes = []
    for line in merged_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip markdown code fence lines
        if line.startswith("```"):
            continue
        try:
            scene = json.loads(line)
            scenes.append(scene)
        except json.JSONDecodeError:
            print(f"  WARNING: Could not parse line: {line[:80]}...")
    return scenes


# ── Output ───────────────────────────────────────────────────────────────────────

def save_gemini_passes(episode_name, pass_results):
    """Save raw Gemini pass results for debugging/reference."""
    ep_dir = GEMINI_OUTPUT_DIR / episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, result) in enumerate(zip(PASS_FILES, pass_results)):
        out_path = ep_dir / name.replace(".txt", "_result.txt")
        out_path.write_text(result, encoding="utf-8")
    print(f"  Saved raw pass results to {ep_dir}")


def apply_scenes_to_manifest(episode_name, scenes):
    """Apply merged scene data to the episode's manifest and caption files."""
    manifest_path = FRAMES_DIR / episode_name / "manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: No manifest found for {episode_name}, skipping apply")
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shots = manifest.get("shots", [])
    applied = 0

    for scene in scenes:
        ts = scene.get("timestamp", "")
        caption = scene.get("caption", "")
        if not caption:
            continue

        # Find matching shot by timestamp
        matched_shot = None
        for shot in shots:
            if shot.get("timestamp_start") == ts:
                matched_shot = shot
                break

        if not matched_shot:
            # Try to match by shot index (scene order = shot order)
            scene_idx = scenes.index(scene)
            if scene_idx < len(shots):
                matched_shot = shots[scene_idx]

        if not matched_shot:
            print(f"  WARNING: No matching shot for timestamp {ts}")
            continue

        shot_num = matched_shot["shot"]
        shot_prefix = f"s{shot_num:02d}"

        # Write caption file
        caption_path = FRAMES_DIR / episode_name / f"{shot_prefix}_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")

        # Update manifest fields
        matched_shot["caption"] = caption
        matched_shot["spy1"] = scene.get("spy1")
        matched_shot["spy2"] = scene.get("spy2")
        matched_shot["spy"] = scene.get("characters", "None")
        matched_shot["view"] = scene.get("framing", "")
        matched_shot["white_hat_off"] = scene.get("white_hat_off", False)
        matched_shot["black_hat_off"] = scene.get("black_hat_off", False)
        matched_shot["nonstandard_outfit"] = scene.get("nonstandard_outfit", False)

        applied += 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Applied {applied}/{len(scenes)} scenes to manifest")
    return applied


# ── Frame Extraction ─────────────────────────────────────────────────────────

def timestamp_to_seconds(ts):
    """Convert M:SS or H:MM:SS timestamp to float seconds."""
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


def extract_frames_for_shot(video_path, start_sec, end_sec, episode_dir, shot_num):
    """Extract frames for a single shot using FFmpeg.

    Uses yadif for deinterlacing, mpdecimate to drop duplicate frames,
    and hue=s=0 for grayscale conversion.
    Returns list of extracted frame paths, or empty list on failure.
    """
    shot_prefix = f"s{shot_num:02d}"
    temp_pattern = str(episode_dir / f"_temp_{shot_prefix}_%04d.jpg")

    # Primary: yadif deinterlace + mpdecimate to drop duplicates
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
        print(f"    WARNING: No frames for shot {shot_num} "
              f"({start_sec}s-{end_sec}s)")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-3:]:
                print(f"      {line}")
        return []

    # Rename sequentially: s01_01.jpg, s01_02.jpg, ...
    renamed = []
    for i, temp_path in enumerate(temp_frames, start=1):
        new_name = episode_dir / f"{shot_prefix}_{i:02d}.jpg"
        temp_path.rename(new_name)
        renamed.append(new_name)

    return renamed


def reextract_episode(episode_name):
    """Re-extract all frames for an episode using merged.jsonl timestamps.

    Reads the merged JSONL, finds the source video, deletes old frames,
    extracts fresh frames at each Gemini scene timestamp, and builds
    a new manifest.
    """
    merged_path = GEMINI_OUTPUT_DIR / episode_name / "merged.jsonl"
    if not merged_path.exists():
        print(f"  SKIP: No merged.jsonl found for {episode_name}")
        return False

    video_path = find_video(episode_name)
    if not video_path:
        print(f"  SKIP: No video found for {episode_name}")
        return False

    # Parse merged scenes
    merged_text = merged_path.read_text(encoding="utf-8")
    scenes = parse_merged_output(merged_text)
    if not scenes:
        print(f"  SKIP: No scenes parsed from merged.jsonl")
        return False

    episode_dir = FRAMES_DIR / episode_name

    # Delete old frames directory
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {len(scenes)} scenes from {video_path.name}...")

    manifest = {"episode": f"{episode_name}{video_path.suffix}", "shots": []}

    for shot_idx, scene in enumerate(scenes):
        shot_num = shot_idx + 1
        ts = scene.get("timestamp", "0:00")
        start_sec = timestamp_to_seconds(ts)

        # End time: next scene's timestamp, or +2s for last scene
        if shot_idx + 1 < len(scenes):
            next_ts = scenes[shot_idx + 1].get("timestamp", "0:00")
            end_sec = timestamp_to_seconds(next_ts)
        else:
            end_sec = start_sec + 2

        # Ensure window is at least 0.5s
        if end_sec <= start_sec:
            end_sec = start_sec + 1

        shot_prefix = f"s{shot_num:02d}"
        caption = scene.get("caption", "")
        spy = scene.get("characters", "None")

        # Add 0.5s padding in both directions to account for Gemini rounding
        # timestamps to whole seconds. This gives more candidate frames in the
        # reviewer without changing the stored timestamps.
        padded_start = max(0, start_sec - 0.5)
        padded_end = end_sec + 0.5

        print(f"    Shot {shot_num:02d} [{ts} -> "
              f"{seconds_to_timestamp(end_sec)}] ({spy})", end="", flush=True)

        frames = extract_frames_for_shot(
            video_path, padded_start, padded_end, episode_dir, shot_num
        )

        selected = None
        if frames:
            # Select midpoint frame
            mid = len(frames) // 2
            selected_frame = frames[mid]
            m = re.match(rf"{shot_prefix}_(\d+)\.jpg", selected_frame.name)
            mid_idx = m.group(1) if m else f"{mid + 1:02d}"

            renamed = selected_frame.with_name(
                f"{shot_prefix}_{mid_idx}_selected.jpg"
            )
            selected_frame.rename(renamed)
            selected = mid_idx
            print(f" - {len(frames)} frames, mid={mid_idx}")
        else:
            print(" - 0 frames")

        # Write caption file
        caption_path = episode_dir / f"{shot_prefix}_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")

        # Build manifest shot entry with all merged metadata
        manifest["shots"].append({
            "shot": shot_num,
            "timestamp_start": ts,
            "timestamp_end": seconds_to_timestamp(end_sec),
            "spy": spy,
            "spy_original": None,
            "caption": caption,
            "candidates": len(frames),
            "selected": selected,
            "excluded": False,
            "white_hat_off": scene.get("white_hat_off", False),
            "black_hat_off": scene.get("black_hat_off", False),
            "nonstandard_outfit": scene.get("nonstandard_outfit", False),
            "appearance_suffix": None,
            "view": scene.get("framing", ""),
            "spy1": scene.get("spy1"),
            "spy2": scene.get("spy2"),
        })

    # Write manifest
    manifest_path = episode_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_frames = sum(s["candidates"] for s in manifest["shots"])
    print(f"  Done: {len(scenes)} shots, {total_frames} total frames")
    return True


def cmd_reextract(episode_name=None):
    """Re-extract frames for episodes using Gemini merged timestamps."""
    if episode_name:
        episodes = [episode_name]
    else:
        # Find all episodes with merged.jsonl
        episodes = []
        if GEMINI_OUTPUT_DIR.exists():
            for d in sorted(GEMINI_OUTPUT_DIR.iterdir()):
                if d.is_dir() and (d / "merged.jsonl").exists():
                    episodes.append(d.name)

    total = len(episodes)
    if total == 0:
        print("No episodes with merged.jsonl found. Run 'run' first.")
        return

    print(f"\nRe-extracting frames for {total} episode(s)...\n")

    success = 0
    for idx, ep in enumerate(episodes, 1):
        print(f"\n[{idx}/{total}] {ep}")
        print(f"{'='*60}")
        try:
            if reextract_episode(ep):
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone! Re-extracted {success}/{total} episode(s).")


# ── Episode Discovery ────────────────────────────────────────────────────────────

def find_video(episode_name):
    """Find video file for an episode name."""
    for ext in (".mp4", ".mkv", ".avi", ".webm"):
        path = EPISODES_DIR / f"{episode_name}{ext}"
        if path.exists():
            return path
    return None


def get_all_episodes():
    """Get list of all episode names from video files."""
    episodes = []
    if EPISODES_DIR.exists():
        for f in sorted(EPISODES_DIR.iterdir()):
            if f.suffix.lower() in (".mp4", ".mkv", ".avi", ".webm"):
                episodes.append(f.stem)
    return episodes


# ── Commands ─────────────────────────────────────────────────────────────────────

def cmd_test(episode_name):
    """Test mode: process one episode, print all results, don't write to manifest."""
    video_path = find_video(episode_name)
    if not video_path:
        print(f"ERROR: Video not found for episode '{episode_name}'")
        sys.exit(1)

    gemini_key = _get_env("GEMINI_API_KEY")
    anthropic_key = _get_env("ANTHROPIC_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    gemini_client = genai.Client(api_key=gemini_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    print(f"\n{'='*60}")
    print(f"TEST MODE: {episode_name}")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")

    # Run Gemini passes
    pass_results = run_all_passes(gemini_client, video_path)

    # Save raw results FIRST (in case display fails)
    save_gemini_passes(episode_name, pass_results)

    # Print raw results
    for i, (name, result) in enumerate(zip(PASS_NAMES, pass_results)):
        print(f"\n{'-'*60}")
        print(f"PASS {i+1}: {name}")
        print(f"{'-'*60}")
        print(result)

    # Merge with Claude
    merged = merge_with_claude(claude_client, pass_results, episode_name)

    print(f"\n{'-'*60}")
    print(f"MERGED OUTPUT (Claude)")
    print(f"{'-'*60}")
    print(merged)

    # Parse and display
    scenes = parse_merged_output(merged)
    print(f"\n{'-'*60}")
    print(f"PARSED SCENES: {len(scenes)}")
    print(f"{'-'*60}")
    for scene in scenes:
        ts = scene.get("timestamp", "?")
        chars = scene.get("characters", "?")
        caption = scene.get("caption", "")
        word_count = len(caption.split())
        print(f"  [{ts}] ({chars}) [{word_count}w] {caption}")

    print(f"\n** Test complete. No files were modified. **")
    print(f"** Raw Gemini results saved to: {GEMINI_OUTPUT_DIR / episode_name} **")
    print(f"** Review the output above. If it looks good, run: **")
    print(f"**   python gemini_captions.py run {episode_name} **")


def cmd_run(episode_name=None):
    """Run mode: process episode(s) and apply results to manifests."""
    gemini_key = _get_env("GEMINI_API_KEY")
    anthropic_key = _get_env("ANTHROPIC_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    gemini_client = genai.Client(api_key=gemini_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    if episode_name:
        episodes = [episode_name]
    else:
        episodes = get_all_episodes()

    total = len(episodes)
    print(f"\nProcessing {total} episode(s)...\n")

    for idx, ep in enumerate(episodes, 1):
        print(f"\n[{idx}/{total}] {ep}")
        print(f"{'='*60}")

        video_path = find_video(ep)
        if not video_path:
            print(f"  SKIP: No video found")
            continue

        # Check if already processed
        pass_dir = GEMINI_OUTPUT_DIR / ep
        merged_path = pass_dir / "merged.jsonl"
        has_passes = pass_dir.exists() and len(list(pass_dir.glob("*_result.txt"))) == 5

        if has_passes and merged_path.exists():
            print(f"  SKIP: Already processed (delete {pass_dir} to reprocess)")
            continue

        try:
            if has_passes:
                # Gemini results exist — just load them and run Claude merge
                print(f"  Found existing Gemini results, running Claude merge only...")
                pass_results = []
                for pf in PASS_FILES:
                    result_path = pass_dir / pf.replace(".txt", "_result.txt")
                    pass_results.append(result_path.read_text(encoding="utf-8"))
            else:
                # Run full Gemini passes
                pass_results = run_all_passes(gemini_client, video_path)
                save_gemini_passes(ep, pass_results)

            # Merge with Claude
            merged = merge_with_claude(claude_client, pass_results, ep)

            # Save merged output
            merged_path.write_text(merged, encoding="utf-8")

            # Parse and apply
            scenes = parse_merged_output(merged)
            apply_scenes_to_manifest(ep, scenes)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone! Processed {total} episode(s).")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-pass Gemini captioning")
    subparsers = parser.add_subparsers(dest="command")

    test_parser = subparsers.add_parser("test", help="Test one episode (no writes)")
    test_parser.add_argument("episode", help="Episode name (e.g., s1_e01_a)")

    run_parser = subparsers.add_parser("run", help="Process and apply to manifests")
    run_parser.add_argument("episode", nargs="?", help="Episode name (omit for all)")

    reextract_parser = subparsers.add_parser(
        "reextract", help="Re-extract frames from Gemini timestamps")
    reextract_parser.add_argument(
        "episode", nargs="?", help="Episode name (omit for all)")

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args.episode)
    elif args.command == "run":
        cmd_run(args.episode)
    elif args.command == "reextract":
        cmd_reextract(args.episode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
