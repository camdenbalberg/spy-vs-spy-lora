"""
Fix hallucinated captions by comparing against voice notes using Claude API.

The voice note is ground truth. Claude edits the generated caption to remove
anything not supported by the voice note.

Usage:
  python fix_captions.py                    # process all (skip s1_0172)
  python fix_captions.py --test s1_0502     # process one frame
  python fix_captions.py --dry-run          # preview without overwriting
"""

import os
import sys
import glob
import re
import time
import argparse

try:
    import anthropic
except ImportError:
    print("Missing: pip install anthropic")
    sys.exit(1)

SELECTED_DIR = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\selected"
SKIP = {"s1_0172"}

# Caption wrappers (same as frame_curator.py)
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

SYSTEM_PROMPT = """You are fixing an AI-generated image caption by comparing it against the user's voice note.

The voice note is ALWAYS correct. It was recorded by a human looking at the image. It may contain filler words (um, uh, actually, no wait, etc.) — ignore those and extract the intended meaning.

The generated caption may contain hallucinations — objects, characters, actions, or settings that the voice note does not mention or confirm. Your job is to:

1. Start with the generated caption as a template for formatting and structure
2. REMOVE anything not supported by the voice note:
   - Extra spies the voice note didn't mention
   - Objects/weapons/props not mentioned
   - Actions not described
   - Indoor/outdoor setting if the voice note says otherwise
3. KEEP details from the generated caption that ARE consistent with the voice note
4. Clean up the voice note's intent into proper caption format
5. Do NOT add anything new. Only use information from the voice note.

Output ONLY the corrected scene description. No preamble."""


def wrap_caption(response_text):
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


def get_api_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                api_key, _ = winreg.QueryValueEx(key, "ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        print("ANTHROPIC_API_KEY not found.")
        sys.exit(1)
    return api_key


def fix_caption(client, voice_text, caption_text):
    user_msg = f"VOICE NOTE:\n{voice_text}\n\nGENERATED CAPTION:\n{caption_text}"
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text


def main():
    parser = argparse.ArgumentParser(description="Fix hallucinated captions")
    parser.add_argument("--test", default=None, help="Process single frame (e.g. s1_0502)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without overwriting")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=get_api_key())

    if args.test:
        bases = [args.test]
    else:
        voice_files = sorted(glob.glob(os.path.join(SELECTED_DIR, "*_voice.txt")),
                             key=lambda x: [int(c) if c.isdigit() else c
                                            for c in re.split(r'(\d+)', os.path.basename(x))])
        bases = [os.path.basename(f).replace("_voice.txt", "") for f in voice_files]

    bases = [b for b in bases if b not in SKIP]
    print(f"Processing {len(bases)} captions...")

    for i, base in enumerate(bases, 1):
        voice_path = os.path.join(SELECTED_DIR, f"{base}_voice.txt")
        caption_path = os.path.join(SELECTED_DIR, f"{base}.txt")

        if not os.path.exists(voice_path):
            print(f"  [{i}/{len(bases)}] {base} — no voice note, skipping")
            continue
        if not os.path.exists(caption_path):
            print(f"  [{i}/{len(bases)}] {base} — no caption, skipping")
            continue

        with open(voice_path, "r") as f:
            voice_text = f.read().strip()
        with open(caption_path, "r") as f:
            caption_text = f.read().strip()

        # Strip existing wrapper to send only the scene description
        scene = caption_text
        if scene.lower().startswith("spyvspy, "):
            scene = scene[len("spyvspy, "):]
        # Remove suffix patterns
        for suf in [SUFFIX_BOTH, SUFFIX_SINGLE, SUFFIX_NONE]:
            if scene.endswith(suf):
                scene = scene[:-len(suf)]
                break

        try:
            fixed = fix_caption(client, voice_text, scene)
            wrapped = wrap_caption(fixed)

            if args.dry_run:
                print(f"\n  [{i}/{len(bases)}] {base}")
                print(f"  VOICE: {voice_text[:80]}...")
                print(f"  OLD:   {caption_text[:80]}...")
                print(f"  NEW:   {wrapped[:80]}...")
            else:
                with open(caption_path, "w") as f:
                    f.write(wrapped)
                print(f"  [{i}/{len(bases)}] {base} — fixed")
        except Exception as e:
            print(f"  [{i}/{len(bases)}] {base} — ERROR: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
