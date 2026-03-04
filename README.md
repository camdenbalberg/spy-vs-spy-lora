# Spy vs Spy — SDXL LoRA Training Pipeline

Training pipeline for an SDXL LoRA that reproduces the Spy vs Spy art style and characters from MAD Magazine / MadTV. Includes automated scene extraction from video, multi-pass AI captioning (Gemini + Claude), a web-based frame reviewer, and kohya_ss training configs.

## Versions

| Version | Images | Source | Key Changes |
|---------|--------|--------|-------------|
| [v1](training/spy-vs-spy-v1/) | 36 | MAD Magazine comic panels | Manual captioning, baseline LoRA |
| [v2](training/spy-vs-spy-v2/) | 220 | v1 panels + MadTV animation frames | Voice-guided Claude Vision captioning, hallucination cleanup pass |
| [v3](training/spy-vs-spy-v3/) | 861 | MadTV animation (DVD rips) | Gemini video captioning, automated frame extraction, PySide6 reviewer |
| [v4](training/spy-vs-spy-v4/) | 946 | MadTV animation (DVD rips) | 5-pass Gemini + Claude merge captioning, web-based reviewer, NovaAnimeXL base |

## General Workflow

Every version follows the same core pipeline. Dataset collection varies by version, but training preparation and execution are consistent:

### 1. Dataset Collection

Gather source images or video. Earlier versions used manually collected comic panels; later versions automated frame extraction from video using FFmpeg with deinterlacing and duplicate removal.

### 2. Captioning

Each training image needs a paired `.txt` caption describing the scene. The captioning approach evolved across versions:

- **v1:** Manual descriptions expanded by AI
- **v2:** Voice notes + Claude Vision, with a second hallucination-cleanup pass
- **v3:** Single-pass Gemini video descriptions
- **v4:** 5-pass Gemini analysis (scene layout, expressions, outfits, objects, backgrounds) merged by Claude into rich per-scene captions

### 3. Caption Formatting

Prose captions are converted to comma-separated tags via the Claude API (`format_captions.py`). Tags are structured as:

```
{trigger_words}, {framing}, {scene tags}, {appearance suffix}
```

Trigger words: `spyvspy` (style), `white_spy` / `black_spy` (characters).

### 4. Manual Review

A frame reviewer UI allows selecting the best candidate frame per shot, editing captions, setting character metadata (spy type, hat state, outfit flags), and excluding bad shots. v3 used a PySide6 desktop app; v4 uses a [FastAPI + vanilla JS web app](training/spy-vs-spy-v4/web/).

### 5. Upscaling

Selected frames are upscaled 4x with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (using the `4x-AnimeSharp` model), then downscaled to 1024px on the longest side for SDXL training.

### 6. Training

LoRA training runs through [kohya_ss sd-scripts](https://github.com/kohya-ss/sd-scripts) with SDXL. Key settings:

| Parameter | Value |
|-----------|-------|
| Network | LoRA, dim=32, alpha=16 |
| Optimizer | AdamW8bit |
| Scheduler | cosine_with_restarts (3 cycles) |
| Precision | bf16 |
| Resolution | 1024, bucketed |
| Epochs | 24, checkpoint every 2 |

Learning rates and base models vary by version — see each version's README for specifics.

### 7. Testing

Trained LoRA checkpoints are loaded in ComfyUI for visual evaluation. Test across multiple epoch checkpoints (early, mid, late) and LoRA strength values (0.6–0.9) to find the sweet spot before overfitting.

## Project Structure

```
├── training/
│   ├── spy-vs-spy-v1/          # 36 comic panels
│   ├── spy-vs-spy-v2/          # 220 mixed images
│   ├── spy-vs-spy-v3/          # 861 animation frames
│   └── spy-vs-spy-v4/          # 946 animation frames (current)
│       ├── gemini_captions.py   # Multi-pass Gemini + Claude captioning
│       ├── format_captions.py   # Claude API prose-to-tags conversion
│       ├── frame_reviewer_web.py # Web-based frame review UI
│       ├── spy_factory.py       # Frame extraction + dataset building
│       ├── prompts/             # Gemini/Claude prompt templates
│       ├── web/                 # Reviewer frontend
│       └── train.toml           # kohya_ss training config
├── docs/                        # Additional documentation
├── find_black_frames.py         # Utility: detect black/empty frames
├── queue_phases.py              # Utility: batch pipeline phases
└── queue_plots.py               # Utility: training loss plots
```

Large files (models, videos, extracted frames, training outputs) are excluded via `.gitignore`. See each version's README for the full directory layout and per-version pipeline details.

## Tools & Dependencies

- [kohya_ss sd-scripts](https://github.com/kohya-ss/sd-scripts) — SDXL LoRA training
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — Image upscaling
- [Gemini API](https://ai.google.dev/) — Video scene analysis
- [Claude API](https://docs.anthropic.com/) — Caption merging, prose-to-tags conversion, vision captioning
- [FFmpeg](https://ffmpeg.org/) — Frame extraction with deinterlacing
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Inference / testing
