# Spy vs Spy LoRA — v2

SDXL LoRA trained on animated frames from the MadTV "Spy vs Spy" TV series, combined with comic panel data from v1.

## Training Data

### New frames (184)

- **Source:** 184 frames hand-selected from ~44,800 extracted animation frames across 2 seasons of the MadTV Spy vs Spy animated series (sourced from YouTube)
- **Selection & captioning:** Used `frame_curator.py`, a custom desktop app that displays frames fullscreen and lets you keep/skip with keyboard controls. When keeping a frame, you record a short voice note describing what's happening. The voice note is transcribed using Google Speech Recognition and sent alongside the frame image to Claude Sonnet (claude-sonnet-4-20250514) with a detailed system prompt (`spyvspy_caption_prompt.md`) that instructs the AI to write a structured scene description using the voice note as ground truth
- **Caption refinement:** After initial captioning, many AI-generated captions contained hallucinated objects, characters, or actions not present in the frames. Ran a second pass (`fix_captions.py`) that sent each voice note + AI caption back to Claude Sonnet as a text-only comparison task — the AI edited the caption to remove anything not supported by the voice note, using the original caption only as a formatting template
- **Upscaling:** Upscaled 4x using Real-ESRGAN (realesrgan-ncnn-vulkan) with the `4x-AnimeSharp` model (selected after A/B testing 13+ upscale models in chaiNNer — AnimeSharp produced the sharpest lines for this art style). Downscaled to 1024px on the longest side for SDXL training
- **Greyscale:** All frames were converted to greyscale to match the black-and-white art style
- **Borders:** Added comic panel borders using `add_borders.py` — a sharpie-and-ruler style with straight lines, slight thickness variation, corner overshoots, rounded endpoints, and asymmetric white padding. This differentiates the new animated frames from the v1 comic panels while adding a comic-like aesthetic

### Reused from v1 (36)

- 36 comic strip panels from v1 (already bordered naturally as comic panels), included as-is with their original captions

### Total: 220 images

## Training Config

- **Base model:** SDXL 1.0 (`sd_xl_base_1.0.safetensors`)
- **Folder:** `2_spyvspy` (2 repeats × 220 images)
- **Epochs:** 24, saving every 2 (12 checkpoints)
- **Batch size:** 2
- **Mixed precision:** bf16 with no_half_vae
- **Optimizer:** AdamW8bit
- **Learning rate:** 1e-4 (UNet), 5e-5 (text encoder)
- **Scheduler:** cosine with restarts (3 cycles), 100 warmup steps
- **Network:** LoRA, dim 32, alpha 16
- **Steps per epoch:** 220, **total steps:** 5,280

## Output

12 checkpoints + final model in `output/`, saved as safetensors. Sweet spot expected around epoch 12–16.

## Tools Used

| Tool | Purpose |
|---|---|
| `frame_curator.py` | Frame selection + voice-guided AI captioning app |
| `fix_captions.py` | Batch hallucination cleanup (voice note vs AI caption) |
| `add_borders.py` | Comic panel border generation (sharpie-and-ruler style) |
| `spyvspy_caption_prompt.md` | System prompt for Claude Vision captioning |
| Real-ESRGAN (4x-AnimeSharp) | Frame upscaling |
| Claude Sonnet | Image captioning + caption refinement |
| Google Speech Recognition | Voice note transcription |

## Known Issues & Future Plans (v3)

- **Blurry source frames:** The animated frames were sourced from a YouTube channel, so even after 4x upscaling with AnimeSharp, some frames still have blurry lines. A v3 should use a higher-quality source (DVD rips, Blu-ray, or direct broadcast captures)
- **Caption quality:** While the two-pass captioning pipeline (AI + hallucination cleanup) improved accuracy significantly, some captions may still have minor issues. Manual review of all 184 captions would further improve quality
