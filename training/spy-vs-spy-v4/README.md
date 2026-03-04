# Spy vs Spy LoRA v4

SDXL LoRA training pipeline for Spy vs Spy style, trained on NovaAnimeXL.

## Changes from v3

- **Base model**: NovaAnimeXL (was SD XL Base 1.0)
- **Three trigger words**: `spyvspy` (style), `white_spy` and `black_spy` (character triggers)
- **No style suffix**: Removed "black and white ink comic art..." — the style is learned from images via `spyvspy`
- **No quality tags**: Removed "masterpiece, best quality"
- **Non-standard outfit support**: Checkbox to skip appearance suffix for disguised/undressed frames
- **Spy type selector**: Dropdown to override White/Black/Both/None per shot
- **Spy override post-processing**: Hands-only close-ups auto-set to `None` spy
- **Lower learning rates**: UNet 5e-5 (was 1e-4), TE 2.5e-5 (was 5e-5)
- **Multi-pass Gemini captioning**: New 5-pass Gemini + Claude merge pipeline replaces single-pass Gemini CSV captions
- **Web-based frame reviewer**: FastAPI + vanilla JS SPA replaces PySide6 desktop app (runs on port 8420)
- **Frame re-extraction from Gemini timestamps**: Frames are extracted at Gemini-identified scene boundaries, not CSV heuristics

## Directory Structure

```
spy-vs-spy-v4/
├── captions/                    # Captioning data
│   ├── gemini-descriptions.csv  # Legacy v3 CSV (no longer primary source)
│   └── gemini_passes/           # Multi-pass Gemini results (v4 source of truth)
│       └── {episode}/
│           ├── pass1_scene_layout_result.txt
│           ├── pass2_expressions_result.txt
│           ├── pass3_outfits_result.txt
│           ├── pass4_objects_result.txt
│           ├── pass5_background_result.txt
│           └── merged.jsonl     # Final merged captions (one JSON object per scene)
├── raw/                         # Source video files
│   ├── episodes/                # Full episodes (.mkv)
│   └── gemini/                  # Short clips for Gemini API (.mp4, <1 min each)
├── frames/                      # Extracted frames + manifests
│   └── {episode}/
│       ├── manifest.json        # Shot metadata (spy type, hat flags, captions, etc.)
│       ├── s01_01.jpg ... s01_NN.jpg   # Candidate frames for shot 1
│       ├── s01_MM_selected.jpg         # Currently selected frame for shot 1
│       └── s01_caption.txt             # Caption text for shot 1
├── prompts/                     # Gemini/Claude prompt templates
│   ├── pass1_scene_layout.txt   # Pass 1: Scene & Layout (establishes timestamps)
│   ├── pass2_expressions.txt    # Pass 2: Expressions & Body Language
│   ├── pass3_outfits.txt        # Pass 3: Outfits & Appearance
│   ├── pass4_objects.txt        # Pass 4: Objects & Props
│   ├── pass5_background.txt     # Pass 5: Background & Environment
│   ├── claude_merge.txt         # Claude merge system prompt
│   └── README.txt               # Prompt system overview
├── web/                         # Web reviewer frontend
│   └── index.html               # Single-page app (vanilla JS)
├── dataset/                     # Image+caption pairs for training
├── upscaled/                    # Real-ESRGAN 4x upscaled images
├── img/1_spyvspy/               # Final training images (1024px)
├── output/                      # Trained LoRA checkpoints
├── log/                         # Training logs
├── gemini_captions.py           # Multi-pass Gemini captioning + frame extraction
├── spy_factory.py               # Legacy frame extraction + dataset building
├── format_captions.py           # Claude API caption formatting (prose -> tags)
├── frame_reviewer.py            # PySide6 review UI (legacy, replaced by web)
├── frame_reviewer_web.py        # FastAPI web-based review UI
├── train.toml                   # kohya_ss training config
└── review_progress.json         # Tracks which episodes have been manually reviewed
```

## Pipeline Overview

### Full Pipeline (in order)

```
Gemini video analysis  →  Claude merge  →  Frame extraction  →  Manual review
→  Caption formatting  →  Dataset build  →  Upscale  →  Downscale  →  Train
```

### Step-by-step

```bash
# 1. Run multi-pass Gemini captioning on all episodes
#    (uploads video to Gemini, runs 5 passes, merges with Claude)
PYTHONIOENCODING=utf-8 python -u gemini_captions.py run

# 2. Re-extract frames at Gemini timestamps
#    (deletes old frames, extracts new ones with yadif deinterlacing + 0.5s padding)
PYTHONIOENCODING=utf-8 python -u gemini_captions.py reextract

# 3. Review frames in web UI (select best candidates, edit captions, set spy types)
python frame_reviewer_web.py
# Open http://localhost:8420

# 4. Format captions via Claude API (prose -> tags)
python format_captions.py run

# 5. Rebuild dataset without API calls (after manual edits)
python format_captions.py rebuild-dataset

# 6. Upscale with Real-ESRGAN 4x
#    (use tools/realesrgan-portable/)

# 7. Downscale to 1024px into img/1_spyvspy/

# 8. Train
cd ../../sd-scripts
source venv/Scripts/activate
PYTHONIOENCODING=utf-8 python sdxl_train_network.py --config_file ../training/spy-vs-spy-v4/train.toml
```

---

## Multi-Pass Gemini Captioning System

### Architecture

Each episode video is sent through 5 focused Gemini prompts, then the results are merged by Claude into a single JSONL caption file. This replaces v3's single-pass Gemini CSV approach.

```
Video → Gemini Pass 1 (Scene & Layout)  ─────────────────────────────────→ ┐
     → Gemini Pass 2 (Expressions)  ─── anchored to Pass 1 timestamps ──→ │
     → Gemini Pass 3 (Outfits)      ─── anchored to Pass 1 timestamps ──→ ├→ Claude Merge → merged.jsonl
     → Gemini Pass 4 (Objects)       ─── anchored to Pass 1 timestamps ──→ │
     → Gemini Pass 5 (Background)    ─── anchored to Pass 1 timestamps ──→ ┘
```

### Pass Details

| Pass | Focus | Output |
|------|-------|--------|
| 1. Scene & Layout | Who, where, action, framing, direction. **Establishes timestamps.** | CSV with timestamp, spies, other_characters, framing, action, spatial, direction |
| 2. Expressions | Emotion, posture, visual cues per character | CSV anchored to Pass 1 timestamps |
| 3. Outfits | Hat on/off, standard vs disguise, outfit description | CSV anchored to Pass 1 timestamps |
| 4. Objects | Held items, interacted items, placed items | CSV anchored to Pass 1 timestamps |
| 5. Background | Setting, location, features, lighting | CSV anchored to Pass 1 timestamps |

Pass 1 is critical — it detects all camera cuts and establishes the scene list. Passes 2-5 receive Pass 1's output and are told "Use these EXACT timestamps — do not add or remove scenes."

### Retry Mechanism

If Pass 1 returns fewer than 6 scenes (constant `PASS1_MIN_SCENES`), the system automatically retries with an augmented prompt telling Gemini it found too few scenes and to re-examine more carefully.

### Models Used

- Gemini: `gemini-2.5-flash`
- Claude merge: `claude-sonnet-4-0`

### API Keys

Set as environment variables:
```bash
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

On Windows, the script also checks User/Machine environment variables via PowerShell if shell env vars aren't set.

### gemini_captions.py Commands

```bash
# Test one episode (prints results, no writes to manifest)
python gemini_captions.py test s1_e01_a

# Process all episodes (runs Gemini passes + Claude merge, writes to manifests)
python gemini_captions.py run

# Process specific episode
python gemini_captions.py run s1_e01_a

# Re-extract frames from merged.jsonl timestamps (all episodes)
python gemini_captions.py reextract

# Re-extract frames for specific episode
python gemini_captions.py reextract s1_e01_a
```

### Skip Logic

- `run`: Skips episodes that already have 5 pass result files AND a merged.jsonl. Delete the episode's `captions/gemini_passes/{episode}/` directory to force re-processing.
- `reextract`: Always re-extracts (deletes and recreates the `frames/{episode}/` directory).

### merged.jsonl Format

Each line is a JSON object:
```json
{
  "timestamp": "0:05",
  "characters": "Black",
  "framing": "full body view",
  "caption": "full body view, black spy sitting on tree branch, scheming expression, holding bomb, large tree, grassy ground",
  "spy1": "black",
  "spy2": null,
  "white_hat_off": false,
  "black_hat_off": false,
  "nonstandard_outfit": false
}
```

---

## Frame Extraction

### How It Works

`gemini_captions.py reextract` reads each episode's `merged.jsonl` and extracts frames using FFmpeg:

1. For each scene, calculates the time window from its timestamp to the next scene's timestamp
2. Adds **0.5 seconds of padding** in both directions to account for Gemini rounding timestamps to whole seconds
3. Extracts all unique frames in that window using:
   - `yadif` — deinterlacing (all frames, not just flagged ones)
   - `mpdecimate` — drops duplicate/near-duplicate frames
   - `hue=s=0` — converts to grayscale
4. Selects the midpoint frame as the default selection
5. Writes caption files and builds `manifest.json`

### FFmpeg Filter Chain

```
yadif,mpdecimate,setpts=N/FRAME_RATE/TB,hue=s=0
```

Falls back to `yadif,hue=s=0` (without mpdecimate) if the first attempt produces 0 frames.

### Timestamp Padding

Gemini rounds timestamps to whole seconds. A scene reported at "0:06" to "0:07" might actually span 6.5s to 7.5s. The 0.5s padding on each side ensures we capture the actual scene content. The padding only affects which candidate frames are extracted — the stored timestamps in the manifest remain as Gemini reported them.

---

## Frame Reviewer (Web)

### Running

```bash
python frame_reviewer_web.py
# Open http://localhost:8420
```

### Features

- Browse all episodes and their shots
- View all candidate frames per shot, select the best one
- Edit captions inline
- Toggle checkboxes: White hat off, Black hat off, Non-standard outfit
- Set spy type: White, Black, Both, None (with editable appearance suffix)
- Exclude shots from the dataset
- Mark episodes as reviewed
- Track review progress via `review_progress.json`

### Data Flow

The reviewer reads from and writes to `frames/{episode}/manifest.json` and `frames/{episode}/s{NN}_caption.txt`. Changes are saved immediately on each action.

---

## manifest.json Schema

Each shot in `manifest.json`:

```json
{
  "shot": 1,
  "timestamp_start": "0:00",
  "timestamp_end": "0:02",
  "spy": "Black",
  "spy_original": null,
  "caption": "full body view, black spy sitting on tree branch...",
  "candidates": 53,
  "selected": "27",
  "excluded": false,
  "white_hat_off": false,
  "black_hat_off": false,
  "nonstandard_outfit": false,
  "appearance_suffix": null,
  "view": "full body view",
  "spy1": "black",
  "spy2": null
}
```

| Field | Description |
|-------|-------------|
| `shot` | 1-indexed shot number |
| `timestamp_start` | Gemini scene start time (M:SS) |
| `timestamp_end` | Next scene's timestamp or +2s for last scene |
| `spy` | Which spy is in the scene: White, Black, Both, None |
| `spy_original` | Original spy value before manual override (null if unchanged) |
| `caption` | Full caption text |
| `candidates` | Number of candidate frames extracted |
| `selected` | Index of the currently selected frame |
| `excluded` | Whether this shot is excluded from the dataset |
| `white_hat_off` | Whether White spy's hat is off |
| `black_hat_off` | Whether Black spy's hat is off |
| `nonstandard_outfit` | Whether any spy is in a non-standard outfit (skip appearance suffix) |
| `appearance_suffix` | Custom appearance suffix override (null = auto-computed) |
| `view` | Framing type from Gemini (full body view, close-up, etc.) |
| `spy1` | Primary spy in the scene (lowercase: "black", "white", or null) |
| `spy2` | Secondary spy if present (lowercase or null) |

---

## Caption Format

### Structure

```
{trigger_prefix}{framing}, {tags}, {appearance_suffix}
```

### Trigger word prefix by spy type

| Spy(s) | Prefix |
|--------|--------|
| White  | `spyvspy, white_spy, ` |
| Black  | `spyvspy, black_spy, ` |
| Both   | `spyvspy, white_spy, black_spy, ` |
| None   | `spyvspy, ` |

### Appearance suffix

- **Single spy (hat on)**: `wearing a fedora hat and trenchcoat with long pointed beak nose and black sclera eyes`
- **Single spy (hat off)**: `hatless, wearing trenchcoat with long pointed beak nose and black sclera eyes`
- **Both spies (hats on)**: `both wearing fedora hats and trenchcoats with long pointed beak noses and black sclera eyes`
- **Both spies (hats off)**: `both hatless, wearing trenchcoats with long pointed beak noses and black sclera eyes`
- **Mixed hat state**: `white spy hatless, black spy wearing fedora hat, both wearing trenchcoats with long pointed beak noses and black sclera eyes`
- **Non-standard outfit**: No appearance suffix
- **None (no spy)**: No appearance suffix

### Spy Override Post-Processing

After Claude formats tags, close-up/extreme close-up shots that contain only disembodied body parts (no face, head, hat, torso keywords) are automatically overridden to `spy: "None"` with spy names stripped from the caption. The original spy value is preserved in `spy_original`.

---

## Format Captions (format_captions.py)

Converts prose captions from Gemini/Claude merge into comma-separated tags suitable for LoRA training using the Claude API.

```bash
# Run conversion on all episodes (skips already-formatted captions)
python format_captions.py run

# Dry run (show what would be converted without API calls)
python format_captions.py dry-run

# Rebuild dataset from existing formatted captions (no API calls)
python format_captions.py rebuild-dataset

# Re-run conversion on a specific episode
python format_captions.py rerun s1_e01_a
```

### Skip Logic

Captions that already have a `_caption_original.txt` backup are considered already formatted and skipped. Delete the `_caption_original.txt` to force re-formatting.

---

## Dataset Building

`format_captions.py rebuild-dataset` or `spy_factory.py dataset` builds the final `dataset/` directory:

1. For each non-excluded shot with a selected frame:
   - Copies the selected frame image to `dataset/`
   - Builds the full caption (trigger prefix + formatted tags + appearance suffix)
   - Writes the caption as a `.txt` file alongside the image

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Base model | NovaAnimeXL |
| Epochs | 24 (save every 2) |
| Batch size | 2 |
| UNet LR | 5e-5 |
| Text Encoder LR | 2.5e-5 |
| Optimizer | AdamW8bit |
| Scheduler | cosine_with_restarts (3 cycles, 100 warmup) |
| Network | LoRA dim=32, alpha=16 |
| Precision | bf16, no_half_vae |
| Resolution | 1024 bucketed 256-2048 step 64 |
| Keep tokens | 1 (protects first trigger word from shuffling) |

### Windows Notes

- `PYTHONIOENCODING=utf-8` required for sd-scripts training to avoid Unicode errors
- ComfyUI needs `--force-fp32-vae` or a baked VAE to avoid black images with SDXL

---

## Generation Settings (ComfyUI)

### Prompt format
```
spyvspy, white_spy, {framing}, {character action/pose}, {environment/setting}, {appearance suffix}
```

### Recommended settings
- Sampler: euler / dpmpp_2m
- Scheduler: normal / karras
- CFG: 5-7
- Steps: 25
- LoRA weight: start at 0.8, test 0.6-0.9
- Clip skip: 2
- Resolution: 1024x1024 or 832x1216

### Negative prompt
```
{character negation}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```

---

## Episode Statistics

- **54 episodes** across 2 seasons
- **946 shots** total (scenes identified by Gemini)
- **57,080 candidate frames** extracted
- Source videos in `raw/gemini/` are short clips (<1 min each) suitable for Gemini API upload

### v3 vs v4 Comparison

v3 used single-pass Gemini CSV captions with 967 total shots. v4's multi-pass system produces 946 shots with significantly richer per-scene metadata (expressions, objects, outfits, background) merged into each caption.

17 episodes that were severely under-segmented in the initial v4 Gemini run were re-processed with an improved Pass 1 prompt that includes a full 25-scene example from s1_e03 and a retry mechanism for low scene counts.
