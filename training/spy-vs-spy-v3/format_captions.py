#!/usr/bin/env python3
"""Convert Gemini prose captions to comma-separated tags for LoRA training.

Uses Claude API to transform verbose prose descriptions into concise,
comma-separated descriptive phrases suitable for LoRA training.

Commands:
  run              Format all captions + rebuild dataset
  dry-run          Show before/after without saving
  rebuild-dataset  Just rebuild dataset/ from current captions (no API calls)
  rerun            Reset an episode's captions from CSV, then re-format
"""

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spy_factory import (
    FRAMES_DIR, DATASET_DIR, build_dataset_for_episode, find_selected_frame,
    build_caption_text,
)

CLAUDE_MODEL = "claude-sonnet-4-0"
MAX_CONCURRENT = 10

CONVERSION_PROMPT = """\
Convert this image caption from prose to comma-separated descriptive tags for LoRA training.

RULES:
1. First tag must be the framing: full body view, close-up, upper body view, extreme close-up, or small figure in the distance
2. Lowercase everything
3. Remove articles (a, an, the) and filler verbs (is, are, appears to be, can be seen)
4. Each tag should be a short descriptive phrase (2-8 words typical)
5. No sentences, no periods — just comma-separated phrases
6. Keep ALL spatial info (left, right, foreground, background, center, through window, beneath table, behind wall)
7. Keep ALL specific objects, body positions, hand actions, expressions, and environment details
8. Do not add any information not in the original caption
9. Do not remove any meaningful details

SPY NAMING RULES (critical):
- If the caption describes a spy's FULL or PARTIAL BODY (face, head, torso, or full figure): keep the spy name (e.g., "black spy leaning forward")
- If the caption ONLY describes disembodied body parts (just hands, just arms, just legs, just feet) with NO mention of face, head, hat, mouth, eyes, eyebrows, torso, standing, sitting, walking, running, or crouching: REMOVE the spy name entirely. Describe generically (e.g., "left hand pulling drawer open" NOT "white spy's left hand pulling drawer open")
- The presence of "right hand" or "left hand" as the ONLY body part mentioned does NOT justify keeping the spy name. The test is: does the caption mention face, head, hat, mouth, eyes, eyebrows, torso, standing, sitting, walking, running, or crouching? If NO, strip the spy name. Period.
- If a spy is only mentioned as a minor background element seen through a window, wall, or obstruction with only limbs visible: REMOVE the spy name. Describe as environment (e.g., "legs visible on bed through window" NOT "black spy's legs on bed through window")

FRAMING OVERRIDE: If you stripped all spy names from the caption, re-evaluate the framing tag. The framing still describes camera distance from the subject, even if the subject is just hands or a room. Use: "extreme close-up" for tight detail shots, "close-up" for hands/objects filling the frame, "medium shot" for partial room/scene views, "wide shot" for full environments or scenes viewed through windows/doorways. Do NOT use "full body view" or "upper body view" when no character body is the subject.

CONTEXT (use ONLY for disambiguation, do NOT pull details from these):
Previous caption: {prev_caption}
Next caption: {next_caption}

CONVERT THIS CAPTION: {caption_text}

Output ONLY the comma-separated tags. Nothing else."""


# -- Core Functions ------------------------------------------------------------


async def convert_caption(client, sem, caption_text, prev_caption=None, next_caption=None):
    """Send a caption to Claude API for prose-to-tags conversion."""
    async with sem:
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": CONVERSION_PROMPT.format(
                    caption_text=caption_text,
                    prev_caption=prev_caption or "None \u2014 first frame",
                    next_caption=next_caption or "None \u2014 last frame",
                ),
            }],
        )
        return response.content[0].text.strip()


def collect_captions(episode_filter=None):
    """Collect all captions that need formatting.

    Skips shots where _caption_original.txt already exists (already formatted).
    Returns list of dicts with episode, shot_prefix, caption_path, text.
    """
    captions = []
    if not FRAMES_DIR.exists():
        return captions

    for ep_dir in sorted(FRAMES_DIR.iterdir()):
        if not ep_dir.is_dir():
            continue
        if episode_filter and ep_dir.name != episode_filter:
            continue

        for caption_path in sorted(ep_dir.glob("s*_caption.txt")):
            shot_prefix = caption_path.stem.replace("_caption", "")
            backup_path = ep_dir / f"{shot_prefix}_caption_original.txt"

            if backup_path.exists():
                continue  # Already formatted in a previous run

            text = caption_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            captions.append({
                "episode": ep_dir.name,
                "shot_prefix": shot_prefix,
                "caption_path": caption_path,
                "backup_path": backup_path,
                "text": text,
            })

    return captions


async def format_captions(episode_filter=None, dry_run=False):
    """Format all captions using Claude API (async, concurrent)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: set ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    captions = collect_captions(episode_filter)
    if not captions:
        print("Nothing to format. All captions already have backups "
              "(_caption_original.txt exists).")
        return set()

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(captions)
    completed = 0

    if dry_run:
        print(f"DRY RUN: showing before/after for {total} captions "
              f"({MAX_CONCURRENT} concurrent)\n")
    else:
        print(f"Formatting {total} captions ({MAX_CONCURRENT} concurrent)...\n")

    # Pre-compute prev/next context for each caption
    contexts = []
    for i, cap in enumerate(captions):
        episode = cap["episode"]
        prev_cap = None
        next_cap = None
        for j in range(i - 1, -1, -1):
            if captions[j]["episode"] == episode:
                prev_cap = captions[j]["text"]
                break
        for j in range(i + 1, len(captions)):
            if captions[j]["episode"] == episode:
                next_cap = captions[j]["text"]
                break
        contexts.append((prev_cap, next_cap))

    # Launch all API calls concurrently (bounded by semaphore)
    async def process_one(i):
        nonlocal completed
        cap = captions[i]
        prev_cap, next_cap = contexts[i]
        try:
            tags = await convert_caption(
                client, sem, cap["text"], prev_cap, next_cap
            )
        except anthropic.APIError as e:
            completed += 1
            print(f"  [{completed}/{total}] {cap['episode']}/{cap['shot_prefix']}"
                  f" - API ERROR: {e}")
            return (i, None)

        completed += 1
        tag_count = len([t for t in tags.split(",") if t.strip()])
        print(f"  [{completed}/{total}] {cap['episode']}/{cap['shot_prefix']}"
              f" - {tag_count} tags")

        if dry_run:
            print(f"    BEFORE: {cap['text'][:120]}...")
            print(f"    AFTER:  {tags[:120]}...")
            print()

        return (i, tags)

    results = await asyncio.gather(*(process_one(i) for i in range(total)))

    # Apply results — file I/O and manifest updates (sequential)
    affected_episodes = set()
    manifest_updates = {}  # episode -> {shot_num: tags}

    for i, tags in results:
        if tags is None:
            continue
        cap = captions[i]
        episode = cap["episode"]

        if not dry_run:
            if not cap["backup_path"].exists():
                cap["backup_path"].write_text(cap["text"], encoding="utf-8")
            cap["caption_path"].write_text(tags, encoding="utf-8")

            shot_num = int(cap["shot_prefix"][1:])
            manifest_updates.setdefault(episode, {})[shot_num] = tags
            affected_episodes.add(episode)

    # Batch-update manifests (one read/write per episode)
    for episode, updates in manifest_updates.items():
        manifest_path = FRAMES_DIR / episode / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for shot in manifest["shots"]:
                if shot["shot"] in updates:
                    shot["caption"] = updates[shot["shot"]]
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    return affected_episodes


def restore_from_csv(episode_filter):
    """Restore original Gemini captions from gemini-descriptions.csv.

    Reads the CSV to find original prose captions, writes them back to
    _caption.txt and _caption_original.txt, and updates manifest.json.
    This undoes any previous formatting so the episode can be re-formatted.
    """
    csv_path = Path(__file__).resolve().parent / "gemini-descriptions.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    # Parse CSV to get episode -> [{timestamp, spy, caption}, ...]
    episode_key = episode_filter + ".mp4"
    csv_captions = {}
    in_episode = False
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row[0].startswith("EPISODE:"):
                in_episode = (episode_key in row[0])
                continue
            if in_episode and len(row) >= 3:
                csv_captions[row[0]] = row[2]

    if not csv_captions:
        print(f"ERROR: No captions found in CSV for {episode_filter}")
        sys.exit(1)

    ep_dir = FRAMES_DIR / episode_filter
    manifest_path = ep_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest found for {episode_filter}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored = 0
    for shot in manifest["shots"]:
        ts = shot["timestamp_start"]
        if ts not in csv_captions:
            print(f"  WARNING: no CSV match for shot {shot['shot']} at {ts}")
            continue

        original_text = csv_captions[ts]
        shot_prefix = f"s{shot['shot']:02d}"

        # Restore caption file to original prose
        caption_path = ep_dir / f"{shot_prefix}_caption.txt"
        caption_path.write_text(original_text, encoding="utf-8")

        # Remove old backup
        backup_path = ep_dir / f"{shot_prefix}_caption_original.txt"
        backup_path.unlink(missing_ok=True)

        # Update manifest
        shot["caption"] = original_text
        restored += 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Restored {restored} captions from CSV for {episode_filter}")
    return restored


def rebuild_dataset(episode_filter=None):
    """Rebuild dataset/ files from current captions."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not FRAMES_DIR.exists():
        print("No frames directory found. Run extract first.")
        return

    total = 0
    for ep_dir in sorted(FRAMES_DIR.iterdir()):
        if not ep_dir.is_dir():
            continue
        if episode_filter and ep_dir.name != episode_filter:
            continue
        count = build_dataset_for_episode(ep_dir.name)
        if count > 0:
            print(f"  {ep_dir.name}: {count} pairs")
        total += count

    print(f"\nDataset rebuilt: {total} image-caption pairs in {DATASET_DIR}")


# -- CLI -----------------------------------------------------------------------


def cmd_run(args):
    episode = getattr(args, "episode", None)
    affected = asyncio.run(format_captions(episode_filter=episode, dry_run=False))
    if affected:
        print(f"\nRebuilding dataset for {len(affected)} affected episodes...")
        for ep in sorted(affected):
            count = build_dataset_for_episode(ep)
            print(f"  {ep}: {count} pairs")
        print("Done.")
    else:
        print("No captions were formatted.")


def cmd_dry_run(args):
    episode = getattr(args, "episode", None)
    asyncio.run(format_captions(episode_filter=episode, dry_run=True))


def cmd_rebuild(args):
    episode = getattr(args, "episode", None)
    rebuild_dataset(episode_filter=episode)


def cmd_rerun(args):
    episode = args.episode
    if not episode:
        print("ERROR: --episode is required for rerun.")
        sys.exit(1)
    print(f"Restoring original captions from CSV for {episode}...")
    restore_from_csv(episode)
    print(f"\nRe-formatting {episode}...")
    affected = asyncio.run(format_captions(episode_filter=episode, dry_run=False))
    if affected:
        print(f"\nRebuilding dataset...")
        for ep in sorted(affected):
            count = build_dataset_for_episode(ep)
            print(f"  {ep}: {count} pairs")
        print("Done.")
    else:
        print("No captions were formatted.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini prose captions to LoRA training tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    p_run = subparsers.add_parser("run", help="Format captions + rebuild dataset")
    p_run.add_argument("--episode", help="Process single episode (e.g. s1_e04_a)")

    p_dry = subparsers.add_parser("dry-run", help="Show before/after without saving")
    p_dry.add_argument("--episode", help="Process single episode (e.g. s1_e04_a)")

    p_rebuild = subparsers.add_parser(
        "rebuild-dataset", help="Rebuild dataset/ from current captions"
    )
    p_rebuild.add_argument("--episode", help="Rebuild single episode")

    p_rerun = subparsers.add_parser(
        "rerun", help="Reset from CSV + re-format (requires --episode)"
    )
    p_rerun.add_argument("--episode", required=True,
                         help="Episode to reset and re-format (e.g. s1_e04_a)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "dry-run": cmd_dry_run,
        "rebuild-dataset": cmd_rebuild,
        "rerun": cmd_rerun,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
