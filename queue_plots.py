"""
Queue efficient XY Plot workflows to ComfyUI for spyvspy_v2 LoRA hyperparameter sweep.

KEY OPTIMIZATION: Each plot uses a FIXED checkpoint + LoRA, and sweeps two
hyperparameters on the X and Y axes. No model swapping within a plot — only
cheap KSampler parameter changes between cells.

3 plot types per (checkpoint, epoch) combo:
  1. LoRA Weight (X) vs CFG (Y)     — 6x5 = 30 images
  2. Sampler (X) vs Scheduler (Y)   — 5x5 = 25 images
  3. Steps (X) vs CFG (Y)           — 6x5 = 30 images

5 checkpoints x 4 epochs x 3 plot types x 3 seeds = 180 plots (~5,100 images)

Queue ordered by checkpoint (outermost) to minimize full model reloads.
"""

import json
import urllib.request

COMFYUI_URL = "http://localhost:8188"

# === Seeds ===
SEEDS = [482917365201, 739156284073, 158294736510]

# === Checkpoints ===
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

# === Sweep values ===
LORA_STRENGTHS = [0.6, 0.75, 0.85, 1.0, 1.25, 1.5]
SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde"]
SCHEDULERS = ["normal", "simple", "karras", "exponential", "sgm_uniform"]

# === Defaults (held constant when not being swept) ===
DEFAULT_LORA_STR = 0.85
DEFAULT_CFG = 7
DEFAULT_STEPS = 25
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# === Prompt / negative ===
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

def make_loader(ckpt_name, lora_stacker_id=None):
    """Eff. Loader SDXL — loads checkpoint, optionally with LoRA stack."""
    inputs = {
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
    }
    if lora_stacker_id:
        inputs["lora_stack"] = [lora_stacker_id, 0]
    return {"class_type": "Eff. Loader SDXL", "inputs": inputs}


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

def make_xy_lora_weights(lora_file, weights):
    """XY Input: LoRA — same LoRA file at different model strengths."""
    inputs = {
        "input_mode": "LoRA Names+Weights",
        "batch_path": "C:\\example_folder",
        "subdirectories": False,
        "batch_sort": "ascending",
        "batch_max": -1,
        "lora_count": len(weights),
        "model_strength": 1,
        "clip_strength": 1,
    }
    for i in range(1, 51):
        if i <= len(weights):
            inputs[f"lora_name_{i}"] = lora_file
            inputs[f"model_str_{i}"] = weights[i - 1]
            inputs[f"clip_str_{i}"] = 1.0
        else:
            inputs[f"lora_name_{i}"] = "None"
            inputs[f"model_str_{i}"] = 1.0
            inputs[f"clip_str_{i}"] = 1.0
    return {"class_type": "XY Input: LoRA", "inputs": inputs}


def make_xy_cfg(first, last, count):
    """XY Input: CFG Scale — linearly spaced values."""
    return {
        "class_type": "XY Input: CFG Scale",
        "inputs": {
            "batch_count": count,
            "first_cfg": float(first),
            "last_cfg": float(last),
        },
    }


def make_xy_steps(first, last, count):
    """XY Input: Steps — linearly spaced step counts."""
    return {
        "class_type": "XY Input: Steps",
        "inputs": {
            "target_parameter": "steps",
            "batch_count": count,
            "first_step": first,
            "last_step": last,
            "first_start_step": 0,
            "last_start_step": 10,
            "first_end_step": 10,
            "last_end_step": 20,
            "first_refine_step": 10,
            "last_refine_step": 20,
        },
    }


def make_xy_sampler(samplers):
    """XY Input: Sampler/Scheduler in sampler-only mode."""
    inputs = {
        "target_parameter": "sampler",
        "input_count": len(samplers),
    }
    for i in range(1, 51):
        if i <= len(samplers):
            inputs[f"sampler_{i}"] = samplers[i - 1]
            inputs[f"scheduler_{i}"] = "None"
        else:
            inputs[f"sampler_{i}"] = "None"
            inputs[f"scheduler_{i}"] = "None"
    return {"class_type": "XY Input: Sampler/Scheduler", "inputs": inputs}


def make_xy_scheduler(schedulers):
    """XY Input: Sampler/Scheduler in scheduler-only mode."""
    inputs = {
        "target_parameter": "scheduler",
        "input_count": len(schedulers),
    }
    for i in range(1, 51):
        if i <= len(schedulers):
            inputs[f"sampler_{i}"] = "None"
            inputs[f"scheduler_{i}"] = schedulers[i - 1]
        else:
            inputs[f"sampler_{i}"] = "None"
            inputs[f"scheduler_{i}"] = "None"
    return {"class_type": "XY Input: Sampler/Scheduler", "inputs": inputs}


# =========================================================================
#  Plot type prompt builders
# =========================================================================

def build_type1(ckpt_name, ckpt_short, lora_file, epoch, seed):
    """Plot Type 1: LoRA Weight (X) vs CFG (Y)  —  6x5 = 30 images.
    Fixed: steps, sampler, scheduler. LoRA applied per-cell by XY Input."""
    title = (f"{ckpt_short} | Ep{epoch} | "
             f"Steps:{DEFAULT_STEPS} | {DEFAULT_SAMPLER} | {DEFAULT_SCHEDULER}")
    prefix = f"spyvspy/2026-02-25/{seed}/{ckpt_short}/svsplot"

    prompt = {}
    prompt["105"] = make_loader(ckpt_name)                       # no lora_stack
    prompt["201"] = make_xy_lora_weights(lora_file, LORA_STRENGTHS)  # X
    prompt["202"] = make_xy_cfg(4.0, 12.0, 5)                       # Y: [4,6,8,10,12]
    prompt["90"]  = make_xy_plot("201", "202", title)
    prompt["106"] = make_ksampler(seed, DEFAULT_STEPS, DEFAULT_CFG,
                                  DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
    prompt["81"]  = make_save_image(prefix)
    return prompt, "LoRA Weight x CFG"


def build_type2(ckpt_name, ckpt_short, lora_file, epoch, seed):
    """Plot Type 2: Sampler (X) vs Scheduler (Y)  —  5x5 = 25 images.
    Fixed: LoRA strength, CFG, steps. LoRA applied via Stacker."""
    title = (f"{ckpt_short} | Ep{epoch} | "
             f"LoRA:{DEFAULT_LORA_STR} | CFG:{DEFAULT_CFG} | Steps:{DEFAULT_STEPS}")
    prefix = f"spyvspy/2026-02-25/{seed}/{ckpt_short}/svsplot"

    prompt = {}
    prompt["200"] = make_lora_stacker(lora_file, DEFAULT_LORA_STR, 1.0)
    prompt["105"] = make_loader(ckpt_name, "200")                    # with lora_stack
    prompt["201"] = make_xy_sampler(SAMPLERS)                        # X
    prompt["202"] = make_xy_scheduler(SCHEDULERS)                    # Y
    prompt["90"]  = make_xy_plot("201", "202", title)
    prompt["106"] = make_ksampler(seed, DEFAULT_STEPS, DEFAULT_CFG,
                                  DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
    prompt["81"]  = make_save_image(prefix)
    return prompt, "Sampler x Scheduler"


def build_type3(ckpt_name, ckpt_short, lora_file, epoch, seed):
    """Plot Type 3: Steps (X) vs CFG (Y)  —  6x5 = 30 images.
    Fixed: LoRA strength, sampler, scheduler. LoRA applied via Stacker."""
    title = (f"{ckpt_short} | Ep{epoch} | "
             f"LoRA:{DEFAULT_LORA_STR} | {DEFAULT_SAMPLER} | {DEFAULT_SCHEDULER}")
    prefix = f"spyvspy/2026-02-25/{seed}/{ckpt_short}/svsplot"

    prompt = {}
    prompt["200"] = make_lora_stacker(lora_file, DEFAULT_LORA_STR, 1.0)
    prompt["105"] = make_loader(ckpt_name, "200")
    prompt["201"] = make_xy_steps(15, 40, 6)                         # X: [15,20,25,30,35,40]
    prompt["202"] = make_xy_cfg(4.0, 12.0, 5)                       # Y: [4,6,8,10,12]
    prompt["90"]  = make_xy_plot("201", "202", title)
    prompt["106"] = make_ksampler(seed, DEFAULT_STEPS, DEFAULT_CFG,
                                  DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
    prompt["81"]  = make_save_image(prefix)
    return prompt, "Steps x CFG"


# =========================================================================
#  Queue management
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


def main():
    builders = [build_type1, build_type2, build_type3]
    builder_labels = ["LoRA Weight x CFG", "Sampler x Scheduler", "Steps x CFG"]

    # Already completed plots (ckpt_short, epoch, builder_index, seed)
    COMPLETED = {
        # sdxl_base Ep14 — all 3 types × all 3 seeds
        ("sdxl_base", 14, 0, 482917365201),
        ("sdxl_base", 14, 0, 739156284073),
        ("sdxl_base", 14, 0, 158294736510),
        ("sdxl_base", 14, 1, 482917365201),
        ("sdxl_base", 14, 1, 739156284073),
        ("sdxl_base", 14, 1, 158294736510),
        ("sdxl_base", 14, 2, 482917365201),
        ("sdxl_base", 14, 2, 739156284073),
        ("sdxl_base", 14, 2, 158294736510),
        # sdxl_base Ep16 — Type1 seed1 only
        ("sdxl_base", 16, 0, 482917365201),
    }

    total_all = len(CHECKPOINTS) * len(LORA_EPOCHS) * len(builders) * len(SEEDS)
    remaining = total_all - len(COMPLETED)

    print(f"Checkpoints:  {len(CHECKPOINTS)}")
    print(f"LoRA epochs:  {len(LORA_EPOCHS)}")
    print(f"Plot types:   3 ({', '.join(builder_labels)})")
    print(f"Seeds:        {len(SEEDS)}")
    print(f"Total plots:  {total_all}")
    print(f"Skipping:     {len(COMPLETED)} (already completed)")
    print(f"Queuing:      {remaining}")
    print(f"Est. images:  ~{remaining * 28}")
    print()

    queued = 0
    skipped = 0
    errors = 0

    # Outer loop: checkpoint (minimizes full model reloads)
    for ckpt_name, ckpt_short in CHECKPOINTS:
        print(f"=== {ckpt_short} ({ckpt_name}) ===")

        for lora_file, epoch in LORA_EPOCHS:
            for bi, builder in enumerate(builders):
                for seed in SEEDS:
                    key = (ckpt_short, epoch, bi, seed)
                    if key in COMPLETED:
                        skipped += 1
                        print(f"  SKIP Ep{epoch} {builder_labels[bi]} seed={seed}")
                        continue

                    prompt, label = builder(ckpt_name, ckpt_short,
                                            lora_file, epoch, seed)
                    result = queue_prompt(prompt)
                    if result and "prompt_id" in result:
                        queued += 1
                        print(f"  [{queued}/{remaining}] Ep{epoch} {label} seed={seed}")
                    else:
                        errors += 1
                        print(f"  FAIL [{queued+errors}/{remaining}] "
                              f"Ep{epoch} {label} seed={seed} -> {result}")

    print(f"\nDone! Queued: {queued}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
