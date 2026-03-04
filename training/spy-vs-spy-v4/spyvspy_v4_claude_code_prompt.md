# Claude Code Prompt: Spy vs Spy LoRA v4 Pipeline Setup

## Overview

Create a v4 training pipeline for the Spy vs Spy LoRA by duplicating and modifying the v3 pipeline. The v3 directory is at `training/spy-vs-spy-v3/`. Create `training/spy-vs-spy-v4/` with updated scripts, config, and directory structure.

## What to copy from v3

Copy these scripts from v3 into v4, then modify them according to the changes below:
- `spy_factory.py` → copy and modify
- `format_captions.py` → copy and modify
- `frame_reviewer.py` → copy and modify
- `train.toml` → copy and modify

Copy these data files/directories as-is (the same source episodes and captions are reused):
- `captions/gemini-descriptions.csv`
- `raw/episodes/` (or symlink — these are the source .mkv files, don't duplicate gigabytes of video)
- `frames/` — copy the ENTIRE directory from v3 including all manifests, selected frames, candidate frames, caption files, and caption_original backups. This preserves all manual review work (hat_off flags, exclusions, selected frames, custom appearance suffixes). Do NOT re-extract.

After copying `frames/`, update every `manifest.json` to add the new `nonstandard_outfit` field (default `false`) to every shot that doesn't already have it. Do not modify any existing fields.

Create these empty directories:
- `dataset/`
- `upscaled/`
- `img/1_spyvspy/`
- `output/`
- `log/`

## Changes from v3

### 1. TRIGGER WORDS (affects spy_factory.py, format_captions.py, frame_reviewer.py)

v3 used a single trigger word: `spyvspy`

v4 uses three trigger words:
- `spyvspy` — style trigger (always present)
- `white_spy` — character trigger (present when White spy is a subject)
- `black_spy` — character trigger (present when Black spy is a subject)

**Caption prefix logic based on Spy(s) column:**
- `White` → `spyvspy, white_spy, `
- `Black` → `spyvspy, black_spy, `
- `Both` → `spyvspy, white_spy, black_spy, `
- `None` → `spyvspy, `

Update `build_caption_text()` in spy_factory.py and the rebuild-dataset logic in format_captions.py accordingly.

The training folder stays `img/1_spyvspy/` — the character triggers are in the caption .txt files only, not the folder name.

### 2. DROP STYLE SUFFIX (affects spy_factory.py, format_captions.py, frame_reviewer.py)

v3 appended this to every caption:
```
, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

v4: **Remove this entirely.** Do not append any style suffix to any caption. The style is learned from the images via the `spyvspy` trigger word.

Also remove `masterpiece, best quality` — not needed for NovaAnimeXL.

### 3. NON-STANDARD OUTFIT SUPPORT (affects frame_reviewer.py, spy_factory.py, format_captions.py)

Add a `nonstandard_outfit` boolean field to manifest.json shots (default: `false`).

In frame_reviewer.py, add a checkbox labeled "Non-standard outfit" for each shot. When checked:
- Set `nonstandard_outfit: true` in manifest
- Skip the appearance suffix entirely for that shot (no fedora, no trenchcoat, no beak nose text)
- The caption just has trigger words + the descriptive tags, nothing about appearance

This is for frames where spies are in disguises, missing clothes, etc. The model still learns poses and composition from these frames.

### 4. APPEARANCE SUFFIX CHANGES (affects spy_factory.py, format_captions.py)

v4 appearance suffix logic (only applied when `nonstandard_outfit` is false):

**Single spy (White or Black):**
- Default: `, wearing a fedora hat and trenchcoat with long pointed beak nose and black sclera eyes`
- If `white_hat_off` or `black_hat_off` (matching spy) is true: `, hatless, wearing trenchcoat with long pointed beak nose and black sclera eyes`

**Both spies:**
- Default: `, both wearing fedora hats and trenchcoats with long pointed beak noses and black sclera eyes`
- If only `white_hat_off`: `, white spy hatless, black spy wearing fedora hat, both wearing trenchcoats with long pointed beak noses and black sclera eyes`
- If only `black_hat_off`: `, white spy wearing fedora hat, black spy hatless, both wearing trenchcoats with long pointed beak noses and black sclera eyes`
- If both hats off: `, both hatless, wearing trenchcoats with long pointed beak noses and black sclera eyes`

**None:** No appearance suffix.

**nonstandard_outfit true:** No appearance suffix regardless of spy type.

### 5. TRAINING CONFIG (train.toml)

Changes from v3:

| Parameter | v3 | v4 |
|-----------|----|----|
| Base model | `sd_xl_base_1.0.safetensors` | Path to NovaAnimeXL checkpoint (user will set this — add a placeholder with comment) |
| Learning rate (UNet) | 1e-4 | 5e-5 |
| Learning rate (TE) | 5e-5 | 2.5e-5 |
| Output prefix | `spyvspy_sdxl_v3` | `spyvspy_sdxl_v4` |
| Output dir | v3 output path | v4 output path |
| Log dir | v3 log path | v4 log path |

Everything else stays the same:
- 24 epochs, save every 2
- batch_size=2
- network_dim=32, network_alpha=16
- AdamW8bit
- cosine_with_restarts (3 cycles, 100 warmup)
- bf16, no_half_vae=true
- resolution 1024 bucketed 256-2048 step 64

### 6. FORMAT_CAPTIONS.PY CONVERSION PROMPT

Replace the existing CONVERSION_PROMPT with:

```python
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
- If a spy is only mentioned as a minor background element seen through a window, wall, or obstruction with only limbs visible: REMOVE the spy name. Describe as environment (e.g., "legs visible on bed through window" NOT "black spy's legs on bed through window")

FRAMING OVERRIDE: If you stripped all spy names from the caption, re-evaluate the framing tag. The framing still describes camera distance from the subject, even if the subject is just hands or a room. Use: 'extreme close-up' for tight detail shots, 'close-up' for hands/objects filling the frame, 'medium shot' for partial room/scene views, 'wide shot' for full environments or scenes viewed through windows/doorways. Do NOT use 'full body view' or 'upper body view' when no character body is the subject.

CONTEXT (use ONLY for disambiguation, do NOT pull details from these):
Previous caption: {prev_caption}
Next caption: {next_caption}

CONVERT THIS CAPTION: {caption_text}

Output ONLY the comma-separated tags. Nothing else."""
```

### 7. SPY OVERRIDE POST-PROCESSING (format_captions.py)

After receiving formatted tags from the API, run this post-processing check:

If the original framing tag is "close-up" or "extreme close-up" AND the caption text does not contain any of these words: face, head, hat, mouth, eyes, eyebrows, body, standing, sitting, walking, running, crouching — then it's a hands-only frame. In that case:
1. Regex replace "white spy's", "black spy's", "white spy", "black spy" with empty string
2. Clean up double commas or leading commas
3. Update manifest.json: set `spy` to `"None"`, preserve original as `"spy_original"`
4. Log: `[OVERRIDE] s1_e03 s04: White → None (hands-only frame)`

### 8. MANIFEST.JSON SCHEMA

Each shot in manifest.json should have these fields:

```json
{
  "shot": 1,
  "timestamp_start": "0:00",
  "timestamp_end": "0:01",
  "spy": "Black",
  "spy_original": null,
  "caption": "raw caption text from Gemini",
  "candidates": 14,
  "selected": "03",
  "selected_score": 1423.7,
  "excluded": false,
  "white_hat_off": false,
  "black_hat_off": false,
  "nonstandard_outfit": false,
  "appearance_suffix": null
}
```

`appearance_suffix`: when null, auto-computed from spy type + hat flags. When set to a string, used as-is (custom override from frame_reviewer).

### 9. FRAME_REVIEWER.PY UI CHANGES

Add to the right panel:

**Spy type selector** (above hat checkboxes):
- Radio buttons or dropdown: White, Black, Both, None
- Pre-populated from manifest's `spy` field
- When changed:
  - Update `spy` in manifest.json
  - Store the original value in `spy_original` if not already set
  - Recompute appearance suffix
  - Recompute caption prefix (trigger words)
  - Update dataset files immediately
  - Log: `[SPY OVERRIDE] s1_e03 s07: Both → White`

**"Non-standard outfit" checkbox** (below hat checkboxes):
- When checked, grays out the hat checkboxes and appearance suffix field (they don't apply)
- When unchecked, hat checkboxes and suffix field are active as normal

Everything else in the reviewer stays the same as v3.

### 10. README.md

Generate a README.md for v4 documenting:
- The pipeline steps (same as v3 but with changes noted)
- Updated prompt format (no style suffix, separate triggers)
- Updated training config differences
- Updated generation settings for NovaAnimeXL
- Directory structure

### Prompt format for v4 generation:
```
spyvspy, white_spy, {framing}, {character action/pose}, {environment/setting}, {appearance suffix}
```

### Recommended v4 generation settings:
- Sampler: euler / dpmpp_2m
- Scheduler: normal / karras
- CFG: 5-7
- Steps: 25
- LoRA weight: start at 0.8, test 0.6-0.9
- Clip skip: 2
- Resolution: 1024x1024 or 832x1216

### Negative prompt:
```
{character negation}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```

Removed from v3 negative: `color, colorful, realistic, photographic, 3d render` — NovaAnimeXL shouldn't need these.

## CLI summary

```bash
# Extract frames from video
python spy_factory.py extract

# Review frames
python frame_reviewer.py

# Format captions via Claude API
python format_captions.py run

# Rebuild dataset (no API calls)
python format_captions.py rebuild-dataset

# Upscale (same as v3)
# Downscale to 1024px into img/1_spyvspy/

# Train
cd ../../sd-scripts
source venv/Scripts/activate
PYTHONIOENCODING=utf-8 python sdxl_train_network.py --config_file ../training/spy-vs-spy-v4/train.toml
```

## Do NOT

- Do not delete or modify anything in the v3 directory
- Do not duplicate the source video files (symlink raw/episodes/ or reference the v3 path)
- Do not change the Gemini captioning prompt or CSV format
- Do not add asyncio where v3 used synchronous code (except format_captions.py which already uses async)
- Do not install new dependencies beyond what v3 already uses
