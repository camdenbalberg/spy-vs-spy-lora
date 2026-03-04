# Spy vs Spy LoRA v3

SDXL LoRA trained on 861 frames from MadTV's Spy vs Spy animated series.

## Quick Start

```bash
# Extract frames from video (skips existing)
python spy_factory.py extract

# Review frames manually
python frame_reviewer.py

# Convert prose captions to tags via Claude API
python format_captions.py run

# Rebuild dataset from current captions (no API calls)
python format_captions.py rebuild-dataset

# Train (after upscale + downscale)
cd ../../sd-scripts
source venv/Scripts/activate
PYTHONIOENCODING=utf-8 python sdxl_train_network.py --config_file ../training/spy-vs-spy-v3/train.toml
```

## Pipeline

1. **Gemini captioning** - Video episodes described by Gemini into `captions/gemini-descriptions.csv`
2. **Frame extraction** - `spy_factory.py extract` pulls frames + builds manifests in `frames/{episode}/`
3. **Manual review** - `frame_reviewer.py` for editing captions, excluding shots, selecting best frames, setting appearance suffixes
4. **Caption formatting** - `format_captions.py run` sends captions to Claude API for prose-to-tag conversion
5. **Dataset build** - `format_captions.py rebuild-dataset` generates image+caption pairs in `dataset/`
6. **Upscale** - Real-ESRGAN 4x anime model (`tools/realesrgan-portable/`)
7. **Downscale** - Resize to 1024px longest side into `img/1_spyvspy/`
8. **Train** - kohya_ss sd-scripts SDXL LoRA

## Scripts

### spy_factory.py

Frame extraction and dataset building.

```bash
python spy_factory.py extract                    # Extract all episodes
python spy_factory.py extract --episode s1_e04_a # Extract single episode
python spy_factory.py dataset                    # Rebuild all dataset pairs
python spy_factory.py dataset --episode s1_e04_a # Rebuild single episode
```

Key functions:
- `compute_appearance_suffix(raw_caption, spy, white_hat_off, black_hat_off)` - Computes appearance text based on spy type and hat state
- `build_caption_text(raw_caption, spy, appearance_suffix=None)` - Assembles final caption: trigger + caption + appearance + style
- `build_dataset_for_episode(episode)` - Writes dataset image+caption pairs from manifest
- `process_episode(episode)` - Extracts video frames (skips if `frames/{episode}/` exists)

### format_captions.py

Claude API prose-to-tags conversion.

```bash
python format_captions.py run                    # Format all unformatted captions
python format_captions.py run --episode s1_e04_a # Format single episode
python format_captions.py dry-run                # Preview without saving
python format_captions.py rebuild-dataset        # Rebuild dataset/ (no API calls)
python format_captions.py rerun --episode s1_e04_a # Reset from CSV + re-format
```

- Skips already-formatted captions (checks for `_caption_original.txt` backup)
- Backs up original caption before overwriting
- Does NOT touch appearance_suffix, excluded, or selected fields
- MAX_CONCURRENT=10 (may hit rate limits; lower to 3 if needed)
- Requires `ANTHROPIC_API_KEY` env var

### frame_reviewer.py

PySide6 desktop app for manual review.

```bash
python frame_reviewer.py
```

**UI features:**
- Left panel: episode list with shot counts and review progress
- Center: selected frame (large), candidate frame strip below
- Right panel: caption editor, spy type, hat-off checkboxes, editable appearance suffix
- Bottom: navigation buttons, exclude toggle

**Controls:**
- Click candidate frames to preview, double-click to select as the dataset frame
- Edit caption text directly (saves to `_caption.txt` and `manifest.json`)
- Toggle "White hat off" / "Black hat off" checkboxes to auto-recompute appearance suffix
- Freely edit the appearance suffix field for custom overrides (e.g., peaked cap, shirtless)
- Exclude/include shots from the training dataset
- All changes persist immediately to manifest and dataset files

**Data flow:**
- Reads from `frames/{episode}/manifest.json`
- Writes to: manifest.json, `s{nn}_caption.txt`, `dataset/{episode}_s{nn}.txt`
- Hat checkbox state stored as `white_hat_off`/`black_hat_off` in manifest
- Custom suffix stored as `appearance_suffix` in manifest

## Directory Structure

```
spy-vs-spy-v3/
├── captions/gemini-descriptions.csv   # Pipeline reads this
├── gemini-descriptions.csv            # Root copy
├── frames/{episode}/                  # Per-episode data
│   ├── manifest.json                  # Shots, captions, metadata
│   ├── s{nn}_caption.txt              # Current caption (tags)
│   ├── s{nn}_caption_original.txt     # Pre-Claude backup
│   ├── s{nn}_mid.jpg                  # Default middle frame
│   └── s{nn}_f{nn}.jpg               # All candidate frames
├── dataset/                           # Image+caption pairs (non-excluded)
├── upscaled/                          # 4x Real-ESRGAN output
├── img/1_spyvspy/                     # Training-ready (1024px + captions)
├── output/                            # LoRA checkpoints
├── log/                               # Training logs
├── spy_factory.py
├── format_captions.py
├── frame_reviewer.py
├── train.toml
└── README.md
```

## Training Config

| Parameter | Value |
|-----------|-------|
| Base model | SDXL 1.0 |
| Images | 861 (1 repeat) |
| Epochs | 24 (saved every 2) |
| Total steps | ~10,344 |
| Batch size | 2 |
| Network | LoRA, dim=32, alpha=16 |
| Optimizer | AdamW8bit |
| Learning rate | 1e-4 (UNet), 5e-5 (text encoder) |
| Scheduler | cosine_with_restarts (3 cycles) |
| Trigger word | `spyvspy` |

## Generation

### Prompt format
```
spyvspy, {framing}, {character action/pose}, {environment/setting}, {appearance suffix}, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style, masterpiece, best quality
```

### Recommended settings
- Sampler: euler / dpmpp_2m
- Scheduler: normal / karras
- CFG: 7
- Steps: 25
- LoRA weight: 0.8

### Negative prompt (Animagine)
```
{character negation}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, color, colorful, realistic, photographic, 3d render
```

### Character negation
- Single white spy prompt: add `black spy, multiple characters, two characters`
- Single black spy prompt: add `white spy, multiple characters, two characters`
- Both spies prompt: add `three characters, extra characters, crowd`

## Notes

- `process_episode()` skips existing episodes; delete `frames/{episode}/` to re-extract
- `format_captions.py` skips already-formatted captions; delete `_caption_original.txt` to re-format
- Use `PYTHONIOENCODING=utf-8` when running sd-scripts on Windows
- ComfyUI: use Baked VAE or `--force-fp32-vae` to avoid black images with SDXL
