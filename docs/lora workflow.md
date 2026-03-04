# SDXL LORA TRAINING WORKFLOW — SPY VS SPY
## Claude Code Phased Execution Guide

### System Context
- **Machine:** camden-desktop (i9-9900K, RTX 3090 24GB, Windows 11)
- **ComfyUI:** C:\ComfyUI-Portable\ComfyUI_windows_portable\
- **Training Tool:** kohya_ss sd-scripts (C:\Users\Camden\Documents\Coding Projects\LoRA training\sd-scripts)
- **Base Model:** SDXL 1.0 (sdxl_base_1.0.safetensors)
- **Goal:** Train a character+style LoRA that produces consistent Spy vs Spy characters (white spy + black spy, long beak noses, fedora hats, trenchcoats, black sclera eyes, slapstick comic art) for 50+ unique Blu-ray disc cover art scenes
- **Trigger Word:** `spyvspy`

---

## PHASE 1: ENVIRONMENT SETUP
**Type: Claude Code executes this**

### 1A. Clone and install kohya sd-scripts

```powershell
cd "C:\Users\Camden\Documents\Coding Projects\LoRA training"
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install xformers
```

### 1B. Create folder structure

```powershell
mkdir C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img
mkdir C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\model
mkdir C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\output
mkdir C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\log
```

### 1C. Download SDXL base model

Download `sd_xl_base_1.0.safetensors` from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

Save to: `C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\model\sd_xl_base_1.0.safetensors`

### 1D. Verify GPU

```powershell
cd "C:\Users\Camden\Documents\Coding Projects\LoRA training\sd-scripts"
.\venv\Scripts\activate
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

Expected: `NVIDIA GeForce RTX 3090` / `VRAM: 24.0 GB`

### ⏸️ CHECKPOINT: Confirm before proceeding
- [ ] torch sees RTX 3090 with 24GB
- [ ] sd-scripts installed without errors
- [ ] SDXL base model downloaded to model folder

---

## PHASE 2: DATASET COLLECTION
**Type: YOU do this manually**

### What to collect: 20-40 images

You're training a **combined character + style LoRA**. The model needs to learn both what the Spy vs Spy characters look like AND the art style they're drawn in.

### Image sourcing strategy

**Priority sources (in order):**
1. **Original Antonio Prohías art** — the classic Mad Magazine strips. This is the definitive style.
2. **Peter Kuper art** — took over Spy vs Spy after Prohías. Slightly different but still iconic. **Pick ONE artist's style, don't mix both** or the LoRA gets confused.
3. **Official Mad Magazine covers** featuring both spies prominently.

**Where to find them:**
- Google Images: `"spy vs spy" prohias comic art`
- Mad Magazine fan archives
- Comic book databases (Comic Vine, Grand Comics Database)
- The Spy vs Spy Wikipedia page has several clean images
- Mad Magazine covers on archive.org

### Image selection rules

**INCLUDE (aim for variety in these):**
- Both spies together in different poses (scheming, fighting, running, sneaking)
- Different scene compositions (indoor, outdoor, close-up, full body)
- Images showing the characters' defining features clearly (beak noses, hats, trenchcoats, black sclera)
- A few solo shots of each spy (maybe 5-8 out of the 30)
- Different gadgets/props/explosions/traps across images

**EXCLUDE:**
- 3D renders, claymation, video game screenshots (wrong style)
- Fan art in different styles (confuses training)
- Panels where characters are tiny or obscured
- Images with heavy text/speech bubbles covering the characters
- Blurry or low-resolution images (minimum 512px on shortest side)
- Images mixing Prohías and Kuper styles in the same dataset

### Image preparation rules

- **Resolution:** Crop/resize to at least 1024×1024 or larger. SDXL native is 1024×1024 but kohya supports bucketing (mixed aspect ratios). Portrait crops fine for cover art practice.
- **Crop tight:** Remove borders, magazine headers, page numbers, unnecessary whitespace. Focus on the characters + scene.
- **Clean up text:** If a panel has a small speech bubble, you can leave it. If text covers >20% of the image, skip it or crop it out.
- **File format:** PNG or JPG, doesn't matter.
- **Naming:** Name files descriptively (helps with captioning later). Example: `both_spies_dynamite_trap_01.png`

### Save all images to:
```
C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\
```

### ⏸️ CHECKPOINT: Tell me when images are collected
- [ ] 20-40 images saved to the img folder
- [ ] All from same artist's style (Prohías OR Kuper, not mixed)
- [ ] Mix of compositions, poses, scenes
- [ ] No 3D/fan art/wrong style mixed in

---

## PHASE 3: CAPTIONING
**Type: Split — Claude helps write, you verify**

### How captioning works

Every training image gets a `.txt` file with the same name. This text tells the model what's in the image, so it learns to associate the trigger word `spyvspy` with the visual style and characters.

**Example:**
```
Image file:  both_spies_dynamite_trap_01.png
Caption file: both_spies_dynamite_trap_01.txt
```

### Caption format

Every caption MUST follow this structure:

```
spyvspy, [character description], [action/scene], [art style descriptors]
```

### Caption writing rules

1. **Always start with the trigger word** `spyvspy`
2. **Describe what's visible** — characters, poses, props, setting
3. **Include consistent style tags** at the end of every caption:
   `black and white ink comic art, bold outlines, high contrast, slapstick cartoon style`
4. **Vary the scene/action description** per image — this is what teaches the model to generalize
5. **Keep captions 1-3 sentences.** Not too short (model won't learn), not too long (overfitting to text)
6. **Don't describe things that aren't in the image** — if there's no explosion, don't say explosion

### Example captions

**Image: Both spies facing off with gadgets**
```
spyvspy, white spy and black spy facing each other, white spy holding a ray gun, black spy hiding dynamite behind his back, both wearing fedora hats and trenchcoats, long pointed beak noses, black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

**Image: White spy sneaking through a door**
```
spyvspy, white spy tiptoeing through a doorway carrying a briefcase, long pointed beak nose, fedora hat, trenchcoat, suspicious expression, black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

**Image: Explosion aftermath**
```
spyvspy, black spy standing triumphantly over a smoking crater, white spy charred and dazed, explosion debris scattered, both characters with long pointed beak noses and fedora hats, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

**Image: Close-up of both spies scheming**
```
spyvspy, close-up of white spy and black spy peering around opposite sides of a wall, beady black sclera eyes, extremely long pointed beak noses, fedora hats, suspicious expressions, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

### How to caption efficiently

**Option A: You write them yourself**
- Open each image, write a caption following the format above
- Save as .txt with matching filename in the same folder

**Option B: Feed images to Claude for captioning (recommended)**
1. Open a separate Claude chat
2. Upload 5-10 images at a time
3. Use this prompt:

```
I'm training an SDXL LoRA for Spy vs Spy characters. For each image I upload, write a training caption following this exact format:

Start with the trigger word "spyvspy", then describe the characters visible (white spy, black spy, or both), their poses and actions, any props/gadgets/traps visible, and the setting. End every caption with: "black and white ink comic art, bold outlines, high contrast, slapstick cartoon style"

Keep each caption 1-3 sentences. Only describe what's actually visible in the image. Output as plain text, one caption per image, labeled by image number.
```

4. Copy captions into .txt files matching each image filename
5. Review every caption — fix anything Claude got wrong about the image content

### Caption file placement

All .txt files go in the SAME folder as the images:
```
C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\
  both_spies_dynamite_01.png
  both_spies_dynamite_01.txt
  white_spy_sneaking_02.png
  white_spy_sneaking_02.txt
  ...
```

### ⏸️ CHECKPOINT: Tell me when captioning is done
- [ ] Every image has a matching .txt file
- [ ] Every caption starts with `spyvspy`
- [ ] Every caption ends with the style tags
- [ ] You've reviewed captions for accuracy

---

## PHASE 4: FOLDER STRUCTURE FOR KOHYA
**Type: Claude Code executes this**

Kohya expects a specific folder naming convention:

```
C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\[repeats]_[concept]\
```

### Claude Code should:

1. **Calculate repeats:**
   - Formula: `repeats = 400 / number_of_images` (rounded to nearest integer)
   - 20 images → 20 repeats
   - 30 images → 13 repeats
   - 40 images → 10 repeats

2. **Create the properly named subfolder and move images + captions into it:**
   ```powershell
   # Example for 30 images:
   mkdir "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\13_spyvspy"
   # Move all images and caption .txt files into this folder
   Move-Item "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\*.png" "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\13_spyvspy\"
   Move-Item "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\*.jpg" "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\13_spyvspy\"
   Move-Item "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\*.txt" "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\13_spyvspy\"
   ```

3. **Verify structure:**
   ```powershell
   # Should show: [N]_spyvspy folder with matched .png/.txt pairs
   Get-ChildItem "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img\*_spyvspy" -Recurse | Group-Object Extension | Select-Object Name, Count
   ```
   Image count and txt count should match.

---

## PHASE 5: TRAINING CONFIGURATION
**Type: Claude Code creates config and runs training**

### Create training config file

Save as `C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\train_config.toml`:

```toml
[model]
pretrained_model_name_or_path = "C:/Users/Camden/Documents/Coding Projects/LoRA training/training/spy-vs-spy/model/sd_xl_base_1.0.safetensors"

[training]
output_dir = "C:/Users/Camden/Documents/Coding Projects/LoRA training/training/spy-vs-spy/output"
output_name = "spyvspy_sdxl"
save_every_n_epochs = 2
max_train_epochs = 16
train_batch_size = 2
learning_rate = 0.0004
text_encoder_lr = 0.00004
lr_scheduler = "cosine"
lr_scheduler_num_cycles = 3
optimizer_type = "AdamW8Bit"
optimizer_args = ["weight_decay=0.1"]
mixed_precision = "bf16"
seed = 42
clip_skip = 1
min_snr_gamma = 5
cache_latents = true
cache_latents_to_disk = false
gradient_checkpointing = true
xformers = true

[network]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[dataset]
resolution = 1024
enable_bucket = true
bucket_no_upscale = true

[logging]
logging_dir = "C:/Users/Camden/Documents/Coding Projects/LoRA training/training/spy-vs-spy/log"
log_with = "tensorboard"
```

### Launch training command

```powershell
cd "C:\Users\Camden\Documents\Coding Projects\LoRA training\sd-scripts"
.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py `
  --pretrained_model_name_or_path="C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\model\sd_xl_base_1.0.safetensors" `
  --train_data_dir="C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\img" `
  --output_dir="C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\output" `
  --output_name="spyvspy_sdxl" `
  --save_every_n_epochs=2 `
  --max_train_epochs=16 `
  --train_batch_size=2 `
  --learning_rate=0.0004 `
  --text_encoder_lr=0.00004 `
  --lr_scheduler="cosine" `
  --lr_scheduler_num_cycles=3 `
  --optimizer_type="AdamW8Bit" `
  --optimizer_args="weight_decay=0.1" `
  --mixed_precision="bf16" `
  --seed=42 `
  --clip_skip=1 `
  --min_snr_gamma=5 `
  --cache_latents `
  --gradient_checkpointing `
  --xformers `
  --network_module="networks.lora" `
  --network_dim=32 `
  --network_alpha=16 `
  --resolution=1024 `
  --enable_bucket `
  --bucket_no_upscale `
  --logging_dir="C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\log" `
  --log_with="tensorboard" `
  --caption_extension=".txt" `
  --shuffle_caption `
  --keep_tokens=1
```

### Key parameter explanations

| Parameter | Value | Why |
|-----------|-------|-----|
| `network_dim` | 32 | Standard for SDXL character+style LoRA. Higher = more capacity but overfitting risk |
| `network_alpha` | 16 | Half of dim — standard scaling factor |
| `learning_rate` | 0.0004 | Proven SDXL rate. Too high = model collapse, too low = undertrained |
| `text_encoder_lr` | 0.00004 | 10x lower than unet LR — text encoder is more sensitive |
| `max_train_epochs` | 16 | Saves every 2 epochs so you get checkpoints at 2,4,6,8,10,12,14,16 |
| `train_batch_size` | 2 | Fits on 3090 24GB easily. Larger batch = smoother gradients |
| `keep_tokens` | 1 | Keeps trigger word `spyvspy` in position 1 always (never shuffled) |
| `shuffle_caption` | yes | Shuffles all tokens AFTER the first one — prevents positional overfitting |
| `min_snr_gamma` | 5 | Stabilizes training, prevents loss spikes |
| `cosine` scheduler | 3 cycles | Learning rate oscillates 3 times — helps escape local minima |
| `enable_bucket` | yes | Allows mixed aspect ratios (don't have to force everything to 1024×1024) |

### Expected training behavior

- **VRAM usage:** ~10-12GB (3090 has plenty of headroom)
- **Training time:** ~20-40 minutes depending on dataset size
- **Output files:** `spyvspy_sdxl-epoch02.safetensors`, `...-epoch04.safetensors`, etc.
- **Power draw:** ~300-400W sustained (same as AI gen workloads)

### ⏸️ CHECKPOINT: Monitor training
- [ ] Training starts without CUDA OOM errors
- [ ] Loss is decreasing over first few epochs
- [ ] Checkpoint files appearing in output folder every 2 epochs
- If loss flatlines early → learning rate may be too low
- If loss spikes or NaN → learning rate too high, reduce to 0.0002
- If CUDA OOM → reduce batch_size to 1

---

## PHASE 6: TESTING THE LORA
**Type: Claude Code sets up, YOU evaluate visually**

### Copy LoRA to ComfyUI

```powershell
# Copy ALL epoch checkpoints for comparison testing
Copy-Item "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\output\*.safetensors" "C:\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\models\loras\"
```

### You also need an SDXL base checkpoint in ComfyUI

If you don't already have one, download `sd_xl_base_1.0.safetensors` to:
```
C:\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\models\checkpoints\
```

### ComfyUI workflow for testing

Build a simple SDXL workflow (NOT the Flux workflow):

1. **CheckpointLoaderSimple** → select `sd_xl_base_1.0.safetensors`
2. **LoraLoader** → select `spyvspy_sdxl-epoch02.safetensors` (start with earliest)
   - `strength_model`: 0.7 (start here, adjust 0.5-1.0)
   - `strength_clip`: 0.7
3. **CLIPTextEncode (positive):**
   ```
   spyvspy, white spy planting a bomb inside a computer server room while black spy watches through security cameras, long pointed beak noses, fedora hats, trenchcoats, black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
   ```
4. **CLIPTextEncode (negative):**
   ```
   blurry, low quality, deformed, realistic, photograph, 3d render, color, watermark, text, signature
   ```
5. **EmptyLatentImage:** 768 × 1152 (portrait for Blu-ray covers)
6. **KSampler:**
   - Steps: 25
   - CFG: 7
   - Sampler: `euler`
   - Scheduler: `normal`
   - Denoise: 1.0
7. **VAEDecode** → **SaveImage**

### Testing strategy

**Test each epoch checkpoint** with the same prompt and seed to compare:
1. Start with epoch 2 — likely underbaked but shows if training direction is right
2. Try epoch 6-8 — usually the sweet spot for SDXL LoRAs
3. Try epoch 14-16 — check for overfitting (signs: frozen poses, identical compositions regardless of prompt, weird artifacts)

**Test LoRA strength:**
- 0.5 — subtle influence, base model still dominant
- 0.7 — good balance (start here)
- 0.9 — strong influence, might overpower prompt
- 1.0 — full strength, risk of artifacts

**Test with different scene prompts** using the best epoch:
```
spyvspy, black spy chasing white spy across rooftops with a giant mallet, ...
spyvspy, both spies disguised as waiters at a fancy restaurant, ...
spyvspy, white spy cracking a safe while black spy sneaks up behind with a pie, ...
```

If characters are recognizable across different scenes → LoRA is working.

### ⏸️ CHECKPOINT: Evaluate results
Tell me:
- [ ] Which epoch checkpoint looks best?
- [ ] Are both spy characters recognizable (beak noses, hats, black/white contrast)?
- [ ] Does changing the scene description actually change the scene?
- [ ] Any artifacts or weird distortions?

---

## PHASE 7: ITERATION (IF NEEDED)
**Type: Claude Code adjusts, you re-evaluate**

### Common problems and fixes

**Problem: Characters look nothing like Spy vs Spy**
- Dataset issue — images too varied in style, or not enough images
- Fix: Curate dataset more strictly, ensure consistent art style
- Fix: Increase network_dim to 64 (more capacity)

**Problem: Every output looks identical (overfitted)**
- Training went too long or learning rate too high
- Fix: Use an earlier epoch checkpoint (epoch 4-8)
- Fix: Reduce max_train_epochs to 10
- Fix: Reduce learning_rate to 0.0002

**Problem: Style is right but characters are inconsistent**
- Captions may not be consistent enough
- Fix: Ensure EVERY caption describes the key character features (beak nose, fedora, trenchcoat, black sclera)
- Fix: Increase keep_tokens to 2-3

**Problem: Artifacts / distortion at higher LoRA strengths**
- Normal for LoRAs above 0.8
- Fix: Use LoRA strength 0.6-0.7
- Fix: Reduce network_dim to 16

**Problem: LoRA only works for scenes in the training data**
- Not enough variety in training images/captions
- Fix: Add more diverse scene compositions to dataset
- Fix: Ensure captions vary the scene descriptions

### Re-training

If adjustments needed, Claude Code modifies the training parameters and reruns Phase 5. Typical iteration:
1. Adjust config
2. Retrain (another 20-40 min)
3. Copy new checkpoints to ComfyUI
4. Test again

---

## PHASE 8: PRODUCTION — GENERATING 50+ COVERS
**Type: Claude Code helps automate, YOU curate**

Once you have a working LoRA + best epoch identified:

### Batch prompt list

Create a list of 50+ unique scene descriptions. Each disc gets a unique spy scenario. Examples themed around encrypted data / espionage:

```
1. white spy cracking an encrypted vault while black spy plants a hidden camera
2. black spy intercepting a secret message from a carrier pigeon while white spy watches from a rooftop
3. both spies racing through a laser security grid in opposite directions
4. white spy disguised as a janitor mopping the floor over a trapdoor while black spy falls in
5. black spy hacking a giant computer while white spy unplugs it from behind
...
```

### ComfyUI batch generation

Use a **Prompt from File** or **CR Prompt List** node to queue all 50+ prompts automatically. Each prompt follows the same structure:

```
spyvspy, [unique scene from list], long pointed beak noses, fedora hats, trenchcoats, black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
```

Generate 3-4 seeds per prompt → cherry-pick the best one per disc.

### Post-processing pipeline

1. **Upscale** best outputs 2-4x using the upscaler models you already have
2. **Open in Photopea** (or batch with ImageMagick)
3. **Add text overlays:** disc number, contents label, spine text
4. **Export at 300 DPI** for print
5. **Print at CopyCorner** on glossy photo paper
6. **Trim and insert** into 7mm slim cases

---

## FILE STRUCTURE SUMMARY

```
C:\Users\Camden\Documents\Coding Projects\LoRA training\
├── sd-scripts\                    (cloned repo + venv)
│   └── venv\
├── training\spy-vs-spy\
│   ├── img\
│   │   └── [N]_spyvspy\          (N = repeats, contains .png + .txt pairs)
│   ├── model\
│   │   └── sd_xl_base_1.0.safetensors
│   ├── output\
│   │   ├── spyvspy_sdxl-epoch02.safetensors
│   │   ├── spyvspy_sdxl-epoch04.safetensors
│   │   └── ...
│   └── log\                       (tensorboard logs)

C:\ComfyUI-Portable\...\ComfyUI\models\
├── checkpoints\
│   └── sd_xl_base_1.0.safetensors
└── loras\
    └── spyvspy_sdxl-epoch08.safetensors  (best checkpoint)
```

---

## CLAUDE CODE LAUNCH PROMPT

Copy this into Claude Code to begin:

```
I need you to set up SDXL LoRA training on my Windows 11 desktop (RTX 3090 24GB).

Follow the workflow in "C:\Users\Camden\Documents\Coding Projects\LoRA training\SDXL_LoRA_Training_Workflow.md"

Start with PHASE 1: Environment Setup.
- Clone kohya-ss/sd-scripts into "C:\Users\Camden\Documents\Coding Projects\LoRA training\sd-scripts"
- Create venv inside sd-scripts, install PyTorch with CUDA 12.4, install requirements + xformers
- Create the folder structure under "C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\"
- Verify GPU detection

Stop after Phase 1 and confirm everything installed correctly before proceeding. I will manually collect training images (Phase 2) and caption them (Phase 3) before you continue.

Do NOT proceed past Phase 1 without my confirmation.
```