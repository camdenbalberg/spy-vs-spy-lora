"""Batch-caption the 36 training images and save results for comparison."""
import os
import sys
import time
import base64
from pathlib import Path

import anthropic

IMG_DIR = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\11_spyvspy"
OUT_DIR = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\test_captions"
PROMPT_FILE = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\spyvspy_caption_prompt.md"

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


def wrap_caption(text):
    t = text.lower()
    has_black = "black spy" in t
    has_white = "white spy" in t
    has_grey = "grey spy" in t or "gray spy" in t
    count = sum([has_black, has_white, has_grey])
    if count >= 2:
        suffix = SUFFIX_BOTH
    elif count == 1:
        suffix = SUFFIX_SINGLE
    else:
        suffix = SUFFIX_NONE
    return PREFIX + text.strip() + suffix


def caption_image(client, prompt, img_path, max_retries=5):
    with open(img_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(img_path).suffix.lower()
    media = "image/png" if ext == ".png" else "image/jpeg"

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.3,
                system=prompt,
                messages=[{
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}],
                }],
            )
            return wrap_caption(msg.content[0].text)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise Exception(f"Failed after {max_retries} retries")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prompt = Path(PROMPT_FILE).read_text(encoding="utf-8")
    client = anthropic.Anthropic()

    images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png"))

    # Skip already-captioned images
    done = set()
    for f in os.listdir(OUT_DIR):
        if f.endswith(".txt"):
            done.add(f.replace(".txt", ".png"))
    remaining = [f for f in images if f not in done]

    print(f"Total: {len(images)}, Already done: {len(done)}, Remaining: {len(remaining)}\n")

    for i, fname in enumerate(remaining, 1):
        path = os.path.join(IMG_DIR, fname)
        caption = caption_image(client, prompt, path)

        # Save generated caption
        out_path = os.path.join(OUT_DIR, Path(fname).stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(caption)

        # Load original caption
        orig_path = os.path.join(IMG_DIR, Path(fname).stem + ".txt")
        original = ""
        if os.path.exists(orig_path):
            with open(orig_path, encoding="utf-8") as f:
                original = f.read().strip()

        print(f"[{len(done) + i}/{len(images)}] {fname}")
        print(f"  GEN: {caption[:150]}...")
        print(f"  ORI: {original[:150]}...")
        print()

        # Pace requests: ~3s between calls to stay under 30k tokens/min
        time.sleep(3)

    print(f"Done! Generated captions saved to: {OUT_DIR}")
    print("Compare side-by-side with your originals.")


if __name__ == "__main__":
    main()
