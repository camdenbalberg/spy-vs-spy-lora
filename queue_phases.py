"""
Phased hyperparameter search for spyvspy_v2 LoRA.

Phase 1: Best checkpoint × epoch  (4×5 = 20 images)
Phase 2: Strength sweep on winners (5×2 = 10 images)  — needs Phase 1 results
Phase 3: Prompt variety on final params (5 single images) — needs Phase 2 results

Single fixed seed across all phases for apples-to-apples comparison.
"""

import json
import urllib.request

COMFYUI_URL = "http://localhost:8188"
SEED = 482917365201

# === Checkpoints (all 5) ===
CHECKPOINTS = [
    ("sd_xl_base_1.0.safetensors", "sdxl_base"),
    ("animagineXL40_v4Opt.safetensors", "animagine"),
    ("counterfeitxl_v25.safetensors", "counterfeit"),
    ("kohaku-xl-zeta.safetensors", "kohaku"),
    ("novaAnimeXL_xlV10.safetensors", "nova"),
]

# === LoRA epochs ===
LORA_EPOCHS = [
    ("spyvspy_v2\\spyvspy_sdxl_v2-000014.safetensors", 14),
    ("spyvspy_v2\\spyvspy_sdxl_v2-000016.safetensors", 16),
    ("spyvspy_v2\\spyvspy_sdxl_v2-000018.safetensors", 18),
    ("spyvspy_v2\\spyvspy_sdxl_v2-000020.safetensors", 20),
]

# === Defaults ===
DEFAULT_LORA_STR = 0.85
DEFAULT_CFG = 7
DEFAULT_STEPS = 25
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# === Prompts ===
POSITIVE = (
    "spyvspy, white spy and black spy fighting over a bomb, "
    "both wearing fedora hats and trenchcoats with long pointed "
    "beak noses and black sclera eyes, explosion in background, "
    "black and white ink comic art, bold outlines, high contrast, "
    " slapstick cartoon style"
)
NEGATIVE = (
    "blurry, low quality, deformed, realistic, photograph, "
    "3d render, color, watermark, text, signature"
)


# =========================================================================
#  Node builders
# =========================================================================

def make_loader(ckpt_name):
    """Eff. Loader SDXL — loads checkpoint (no LoRA stack; XY handles LoRA)."""
    return {
        "class_type": "Eff. Loader SDXL",
        "inputs": {
            "base_ckpt_name": ckpt_name,
            "base_clip_skip": -2,
            "refiner_ckpt_name": "None",
            "refiner_clip_skip": -2,
            "positive_ascore": 6,
            "negative_ascore": 2,
            "vae_name": "sdxlVae.jwfj.safetensors",
            "positive": POSITIVE,
            "negative": NEGATIVE,
            "token_normalization": "none",
            "weight_interpretation": "comfy",
            "empty_latent_width": 960,
            "empty_latent_height": 1152,
            "batch_size": 1,
        },
    }


def make_loader_with_lora(ckpt_name, lora_stacker_id):
    """Eff. Loader SDXL with LoRA stack connected."""
    loader = make_loader(ckpt_name)
    loader["inputs"]["lora_stack"] = [lora_stacker_id, 0]
    return loader


def make_lora_stacker(lora_file, model_str=0.85, clip_str=1.0):
    """LoRA Stacker — applies a single LoRA at fixed strength."""
    inputs = {
        "input_mode": "advanced",
        "lora_count": 1,
    }
    for i in range(1, 51):
        if i == 1:
            inputs[f"lora_name_{i}"] = lora_file
            inputs[f"lora_wt_{i}"] = model_str
            inputs[f"model_str_{i}"] = model_str
            inputs[f"clip_str_{i}"] = clip_str
        else:
            inputs[f"lora_name_{i}"] = "None"
            inputs[f"lora_wt_{i}"] = 1.0
            inputs[f"model_str_{i}"] = 1.0
            inputs[f"clip_str_{i}"] = 1.0
    return {"class_type": "LoRA Stacker", "inputs": inputs}


def make_ksampler(seed, steps, cfg, sampler, scheduler):
    """KSampler SDXL (Eff.)"""
    return {
        "class_type": "KSampler SDXL (Eff.)",
        "inputs": {
            "sdxl_tuple": ["105", 0],
            "latent_image": ["105", 1],
            "optional_vae": ["105", 2],
            "script": ["90", 0],
            "noise_seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "start_at_step": 0,
            "refine_at_step": -1,
            "preview_method": "none",
            "vae_decode": "true",
        },
    }


def make_xy_plot(x_node_id, y_node_id, title):
    """XY Plot node."""
    return {
        "class_type": "XY Plot",
        "inputs": {
            "dependencies": ["105", 3],
            "X": [x_node_id, 0],
            "Y": [y_node_id, 0],
            "grid_spacing": 15,
            "XY_flip": "False",
            "Y_label_orientation": "Horizontal",
            "cache_models": "False",
            "ksampler_output_image": "Plot",
            "plot_title": title,
        },
    }


def make_save_image(prefix):
    return {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["106", 3],
            "filename_prefix": prefix,
        },
    }


# =========================================================================
#  XY Input builders
# =========================================================================

def make_xy_lora_epochs(lora_files_and_epochs, model_str=0.85):
    """XY Input: LoRA — different epoch files at fixed strength."""
    count = len(lora_files_and_epochs)
    inputs = {
        "input_mode": "LoRA Names+Weights",
        "batch_path": "C:\\example_folder",
        "subdirectories": False,
        "batch_sort": "ascending",
        "batch_max": -1,
        "lora_count": count,
        "model_strength": 1,
        "clip_strength": 1,
    }
    for i in range(1, 51):
        if i <= count:
            inputs[f"lora_name_{i}"] = lora_files_and_epochs[i - 1][0]
            inputs[f"model_str_{i}"] = model_str
            inputs[f"clip_str_{i}"] = 1.0
        else:
            inputs[f"lora_name_{i}"] = "None"
            inputs[f"model_str_{i}"] = 1.0
            inputs[f"clip_str_{i}"] = 1.0
    return {"class_type": "XY Input: LoRA", "inputs": inputs}


def make_xy_lora_strengths(lora_file, strengths):
    """XY Input: LoRA — same file at different strengths."""
    inputs = {
        "input_mode": "LoRA Names+Weights",
        "batch_path": "C:\\example_folder",
        "subdirectories": False,
        "batch_sort": "ascending",
        "batch_max": -1,
        "lora_count": len(strengths),
        "model_strength": 1,
        "clip_strength": 1,
    }
    for i in range(1, 51):
        if i <= len(strengths):
            inputs[f"lora_name_{i}"] = lora_file
            inputs[f"model_str_{i}"] = strengths[i - 1]
            inputs[f"clip_str_{i}"] = 1.0
        else:
            inputs[f"lora_name_{i}"] = "None"
            inputs[f"model_str_{i}"] = 1.0
            inputs[f"clip_str_{i}"] = 1.0
    return {"class_type": "XY Input: LoRA", "inputs": inputs}


def make_xy_checkpoints(checkpoint_list):
    """XY Input: Checkpoint — sweep base checkpoints with clip_skip=-2."""
    count = len(checkpoint_list)
    inputs = {
        "target_ckpt": "Base",
        "input_mode": "Ckpt Names+ClipSkip+VAE",
        "batch_path": "C:\\example_folder",
        "subdirectories": False,
        "batch_sort": "ascending",
        "batch_max": -1,
        "ckpt_count": count,
    }
    for i in range(1, 51):
        if i <= count:
            inputs[f"ckpt_name_{i}"] = checkpoint_list[i - 1][0]
            inputs[f"clip_skip_{i}"] = -2
            inputs[f"vae_name_{i}"] = "sdxlVae.jwfj.safetensors"
        else:
            inputs[f"ckpt_name_{i}"] = "None"
            inputs[f"clip_skip_{i}"] = -1
            inputs[f"vae_name_{i}"] = "Baked VAE"
    return {"class_type": "XY Input: Checkpoint", "inputs": inputs}


# =========================================================================
#  Queue helper
# =========================================================================

def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# =========================================================================
#  Phase 1: Best checkpoint × epoch
# =========================================================================

def phase1():
    """4 LoRA epochs (X) × 5 checkpoints (Y) = 20 images.
    Fixed: LoRA strength 0.85, CFG 7, steps 25, euler, simple."""

    print("=" * 60)
    print("PHASE 1: Best checkpoint × epoch (4×5 = 20 images)")
    print("=" * 60)
    print(f"  X: LoRA epochs {[e for _, e in LORA_EPOCHS]}")
    print(f"  Y: Checkpoints {[s for _, s in CHECKPOINTS]}")
    print(f"  Fixed: LoRA:{DEFAULT_LORA_STR} CFG:{DEFAULT_CFG} "
          f"Steps:{DEFAULT_STEPS} {DEFAULT_SAMPLER} {DEFAULT_SCHEDULER}")
    print(f"  Seed: {SEED}")
    print()

    title = (f"Phase 1: Epoch x Checkpoint | LoRA:{DEFAULT_LORA_STR} | "
             f"CFG:{DEFAULT_CFG} | Steps:{DEFAULT_STEPS} | "
             f"{DEFAULT_SAMPLER} | {DEFAULT_SCHEDULER}")
    prefix = "spyvspy/phases/phase1"

    prompt = {}
    prompt["105"] = make_loader(CHECKPOINTS[0][0])  # default ckpt (will be overridden)
    prompt["201"] = make_xy_lora_epochs(LORA_EPOCHS, DEFAULT_LORA_STR)  # X
    prompt["202"] = make_xy_checkpoints(CHECKPOINTS)                     # Y
    prompt["90"]  = make_xy_plot("201", "202", title)
    prompt["106"] = make_ksampler(SEED, DEFAULT_STEPS, DEFAULT_CFG,
                                  DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
    prompt["81"]  = make_save_image(prefix)

    result = queue_prompt(prompt)
    if result and "prompt_id" in result:
        print(f"  Queued! prompt_id={result['prompt_id']}")
    else:
        print(f"  FAILED: {result}")


# =========================================================================
#  Phase 2: Strength sweep on winners
# =========================================================================

# Phase 1 results: sdxl_base and novaAnime won, Ep14 best overall
PHASE2_CKPTS = [
    ("sd_xl_base_1.0.safetensors", "sdxl_base"),
    ("novaAnimeXL_xlV10.safetensors", "nova"),
]
PHASE2_LORA = "spyvspy_v2\\spyvspy_sdxl_v2-000014.safetensors"
PHASE2_STRENGTHS = [0.5, 0.7, 0.85, 1.0, 1.25]


def phase2():
    """5 LoRA strengths (X) × 2 checkpoints (Y) = 10 images.
    Fixed: Ep14, CFG 7, steps 25, euler, simple."""

    print("=" * 60)
    print("PHASE 2: Strength sweep on winners (5×2 = 10 images)")
    print("=" * 60)
    print(f"  X: LoRA strengths {PHASE2_STRENGTHS}")
    print(f"  Y: Checkpoints {[s for _, s in PHASE2_CKPTS]}")
    print(f"  Fixed: Ep14 CFG:{DEFAULT_CFG} "
          f"Steps:{DEFAULT_STEPS} {DEFAULT_SAMPLER} {DEFAULT_SCHEDULER}")
    print(f"  Seed: {SEED}")
    print()

    title = (f"Phase 2: LoRA Strength x Checkpoint | Ep14 | "
             f"CFG:{DEFAULT_CFG} | Steps:{DEFAULT_STEPS} | "
             f"{DEFAULT_SAMPLER} | {DEFAULT_SCHEDULER}")
    prefix = "spyvspy/phases/phase2"

    prompt = {}
    prompt["105"] = make_loader(PHASE2_CKPTS[0][0])
    prompt["201"] = make_xy_lora_strengths(PHASE2_LORA, PHASE2_STRENGTHS)  # X
    prompt["202"] = make_xy_checkpoints(PHASE2_CKPTS)                       # Y
    prompt["90"]  = make_xy_plot("201", "202", title)
    prompt["106"] = make_ksampler(SEED, DEFAULT_STEPS, DEFAULT_CFG,
                                  DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
    prompt["81"]  = make_save_image(prefix)

    result = queue_prompt(prompt)
    if result and "prompt_id" in result:
        print(f"  Queued! prompt_id={result['prompt_id']}")
    else:
        print(f"  FAILED: {result}")


# =========================================================================
#  Phase 3: Prompt variety on final params
# =========================================================================

# Phase 2 result: 0.85 strength wins
PHASE3_LORA = "spyvspy_v2\\spyvspy_sdxl_v2-000014.safetensors"
PHASE3_STR = 0.85
PHASE3_CKPTS = [
    ("sd_xl_base_1.0.safetensors", "sdxl_base"),
    ("novaAnimeXL_xlV10.safetensors", "nova"),
]

PHASE3_PROMPTS = {
    "P1_fight": (
        "spyvspy, white spy and black spy fighting over a bomb, "
        "both wearing fedora hats and trenchcoats with long pointed "
        "beak noses and black sclera eyes, explosion in background, "
        "black and white ink comic art, bold outlines, high contrast, "
        " slapstick cartoon style"
    ),
    "P2_dynamite": (
        "spyvspy, black spy holding a dynamite stick with a lit fuse, "
        "sneaking through a doorway, wearing a fedora hat and trenchcoat "
        "with long pointed beak nose and black sclera eyes, black and "
        "white ink comic art, bold outlines, high contrast, slapstick "
        "cartoon style"
    ),
    "P3_jungle": (
        "spyvspy, white spy hiding behind a tree in a jungle, wearing a "
        "fedora hat and trenchcoat with long pointed beak nose and black "
        "sclera eyes, black and white ink comic art, bold outlines, high "
        "contrast, slapstick cartoon style"
    ),
    "P4_trap": (
        "spyvspy, white spy falling into a trap hole in the floor, arms "
        "flailing, wearing a fedora hat and trenchcoat with long pointed "
        "beak nose and black sclera eyes, black and white ink comic art, "
        "bold outlines, high contrast, slapstick cartoon style"
    ),
    "P5_backtoback": (
        "spyvspy, white spy and black spy standing back to back holding "
        "weapons, both wearing fedora hats and trenchcoats with long "
        "pointed beak noses and black sclera eyes, dramatic lighting, "
        "black and white ink comic art, bold outlines, high contrast, "
        "slapstick cartoon style"
    ),
}


def phase3():
    """5 prompts × 2 checkpoints = 10 single images.
    Fixed: Ep14, LoRA 0.85, CFG 7, steps 25, euler, simple."""

    print("=" * 60)
    print("PHASE 3: Prompt variety (5 prompts × 2 checkpoints = 10 images)")
    print("=" * 60)
    print(f"  Checkpoints: {[s for _, s in PHASE3_CKPTS]}")
    print(f"  Fixed: Ep14 LoRA:{PHASE3_STR} CFG:{DEFAULT_CFG} "
          f"Steps:{DEFAULT_STEPS} {DEFAULT_SAMPLER} {DEFAULT_SCHEDULER}")
    print(f"  Seed: {SEED}")
    print()

    queued = 0
    for ckpt_name, ckpt_short in PHASE3_CKPTS:
        for prompt_key, positive in PHASE3_PROMPTS.items():
            prefix = f"spyvspy/phases/phase3/{ckpt_short}_{prompt_key}"

            p = {}
            p["200"] = make_lora_stacker(PHASE3_LORA, PHASE3_STR, 1.0)
            p["105"] = make_loader_with_lora(ckpt_name, "200")
            p["106"] = {
                "class_type": "KSampler SDXL (Eff.)",
                "inputs": {
                    "sdxl_tuple": ["105", 0],
                    "latent_image": ["105", 1],
                    "optional_vae": ["105", 2],
                    "noise_seed": SEED,
                    "steps": DEFAULT_STEPS,
                    "cfg": DEFAULT_CFG,
                    "sampler_name": DEFAULT_SAMPLER,
                    "scheduler": DEFAULT_SCHEDULER,
                    "start_at_step": 0,
                    "refine_at_step": -1,
                    "preview_method": "none",
                    "vae_decode": "true",
                },
            }
            p["81"] = {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["106", 3],
                    "filename_prefix": prefix,
                },
            }

            # Override the positive prompt in the loader
            p["105"]["inputs"]["positive"] = positive

            result = queue_prompt(p)
            if result and "prompt_id" in result:
                queued += 1
                print(f"  [{queued}/10] {ckpt_short} {prompt_key}")
            else:
                print(f"  FAIL {ckpt_short} {prompt_key} -> {result}")

    print(f"\nDone! Queued: {queued}/10")


# =========================================================================
#  Main
# =========================================================================

if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"

    if phase == "1":
        phase1()
    elif phase == "2":
        phase2()
    elif phase == "3":
        phase3()
    else:
        print(f"Unknown phase: {phase}")
        print("Usage: python queue_phases.py [1|2|3]")
