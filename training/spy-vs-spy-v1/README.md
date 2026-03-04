# Spy vs Spy LoRA — v1

SDXL LoRA trained on the "Spy vs Spy" comic strip art from MAD Magazine.

## Training Data

- **Source:** 36 comic strip panels from the MAD Magazine "Spy vs Spy" series
- **Upscaling:** Upscaled to 4K using Real-ESRGAN (realesrgan-ncnn-vulkan, x4plus-anime model), then downscaled to 1024px on the longest side for SDXL training
- **Captioning:** Descriptions were manually typed, then fed into AI to expand with additional relevant detail. All descriptions were manually reviewed and tweaked for curation quality
- **Format:** Each panel paired with a `.txt` caption file in kohya DreamBooth format

## Training Config

- **Base model:** SDXL 1.0 (`sd_xl_base_1.0.safetensors`)
- **Folder:** `11_spyvspy` (11 repeats × 36 images)
- **Epochs:** 24, saving every 2 (12 checkpoints)
- **Batch size:** 2
- **Optimizer:** AdamW8bit
- **Learning rate:** 1e-4 (UNet), 5e-5 (text encoder)
- **Scheduler:** cosine with restarts (3 cycles)
- **Network:** LoRA, dim 32, alpha 16

## Output

12 checkpoints in `output/`, saved as safetensors.

## Notes

- A greyscaled variant of the training data exists (`11_spyvspy - greyscaled`). A future version could retrain with the greyscale filter forced on all panels to better match the original black-and-white ink art style
- Best checkpoint from v1 testing was around epoch 10
