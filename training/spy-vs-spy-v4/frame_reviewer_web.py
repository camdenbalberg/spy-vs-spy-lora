#!/usr/bin/env python3
"""Frame Reviewer Web - FastAPI backend for reviewing Spy vs Spy LoRA training frames.

Mirrors the PySide6 frame_reviewer.py functionality as a REST API.
Both can run side-by-side — they share the same manifest.json and caption files.

Usage:
    pip install fastapi uvicorn
    python frame_reviewer_web.py
    # Open http://localhost:8420 on your phone
"""

import json
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ── Paths ────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
FRAMES_DIR = PROJECT_ROOT / "frames"
DATASET_DIR = PROJECT_ROOT / "dataset"
REVIEW_PROGRESS_PATH = PROJECT_ROOT / "review_progress.json"
WEB_DIR = PROJECT_ROOT / "web"

# ── Constants (mirrored from spy_factory.py) ─────────────────────────────────────

TRIGGER_STYLE = "spyvspy"
TRIGGER_WHITE = "white_spy"
TRIGGER_BLACK = "black_spy"

APPEARANCE_SINGULAR = (
    "wearing a fedora hat and trenchcoat with long pointed beak nose "
    "and black sclera eyes"
)
APPEARANCE_SINGULAR_HATLESS = (
    "hatless, wearing trenchcoat with long pointed beak nose "
    "and black sclera eyes"
)
APPEARANCE_PLURAL = (
    "both wearing fedora hats and trenchcoats with long pointed beak noses "
    "and black sclera eyes"
)
APPEARANCE_PLURAL_BOTH_HATLESS = (
    "both hatless, wearing trenchcoats with long pointed beak noses "
    "and black sclera eyes"
)

VIEW_OPTIONS = [
    "full body view", "upper body view", "close-up", "wide shot",
    "medium shot", "extreme close-up", "small figure in the distance",
    "side view", "overhead view", "low angle view", "split screen",
]


# ── Caption / Appearance Helpers (duplicated to avoid PySide6 import) ────────────

def build_trigger_prefix(spy):
    spy_lower = (spy or "None").lower()
    if spy_lower == "white":
        return f"{TRIGGER_STYLE}, {TRIGGER_WHITE}, "
    elif spy_lower == "black":
        return f"{TRIGGER_STYLE}, {TRIGGER_BLACK}, "
    elif spy_lower == "both":
        return f"{TRIGGER_STYLE}, {TRIGGER_WHITE}, {TRIGGER_BLACK}, "
    return f"{TRIGGER_STYLE}, "


def compute_appearance_suffix(spy, white_hat_off=False, black_hat_off=False,
                              nonstandard_outfit=False):
    if nonstandard_outfit:
        return ""
    spy_lower = (spy or "").lower()
    if spy_lower in ("white", "black"):
        hat_off = white_hat_off if spy_lower == "white" else black_hat_off
        return APPEARANCE_SINGULAR_HATLESS if hat_off else APPEARANCE_SINGULAR
    elif spy_lower == "both":
        if white_hat_off and black_hat_off:
            return APPEARANCE_PLURAL_BOTH_HATLESS
        elif not white_hat_off and not black_hat_off:
            return APPEARANCE_PLURAL
        else:
            w = "hatless" if white_hat_off else "wearing fedora hat"
            b = "hatless" if black_hat_off else "wearing fedora hat"
            return (
                f"white spy {w}, black spy {b}, "
                f"both wearing trenchcoats with long pointed beak noses "
                f"and black sclera eyes"
            )
    return ""


def build_caption_text(raw_caption, spy, appearance_suffix=None,
                       nonstandard_outfit=False):
    prefix = build_trigger_prefix(spy)
    text = f"{prefix}{raw_caption}"
    if appearance_suffix is None:
        appearance_suffix = compute_appearance_suffix(spy, nonstandard_outfit=nonstandard_outfit)
    if appearance_suffix:
        text += f", {appearance_suffix}"
    return text


def parse_view_from_caption(caption_text):
    comma_idx = caption_text.find(",")
    if comma_idx > 0:
        return caption_text[:comma_idx].strip()
    return ""


def parse_spies_from_caption(caption_text):
    lower = caption_text.lower()
    white_pos = lower.find("white spy")
    black_pos = lower.find("black spy")
    if white_pos >= 0 and black_pos >= 0:
        if white_pos < black_pos:
            return ("white", "black")
        return ("black", "white")
    elif white_pos >= 0:
        return ("white", None)
    elif black_pos >= 0:
        return ("black", None)
    return (None, None)


def derive_spy_type(spy1, spy2):
    has1 = spy1 is not None
    has2 = spy2 is not None
    if has1 and has2:
        return "Both"
    elif has1:
        return spy1.capitalize()
    elif has2:
        return spy2.capitalize()
    return "None"


def replace_spy_in_caption(caption, old_color, new_color):
    return re.sub(
        rf'{re.escape(old_color)} spy', f'{new_color} spy',
        caption, flags=re.IGNORECASE,
    )


def swap_spies_in_caption(caption, color_a, color_b):
    temp = re.sub(rf'{re.escape(color_a)} spy', '\x00SPY_A\x00',
                  caption, flags=re.IGNORECASE)
    temp = re.sub(rf'{re.escape(color_b)} spy', '\x00SPY_B\x00',
                  temp, flags=re.IGNORECASE)
    result = temp.replace('\x00SPY_A\x00', f'{color_b} spy')
    return result.replace('\x00SPY_B\x00', f'{color_a} spy')


def strip_spy_names(caption_text):
    cleaned = re.sub(r"white spy'?s?\b", "", caption_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"black spy'?s?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


# ── Manifest Helpers ─────────────────────────────────────────────────────────────

def _read_manifest(episode):
    manifest_path = FRAMES_DIR / episode / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Episode '{episode}' not found")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest(episode, manifest):
    manifest_path = FRAMES_DIR / episode / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _find_shot(manifest, shot_num):
    for s in manifest["shots"]:
        if s["shot"] == shot_num:
            return s
    raise HTTPException(status_code=404, detail=f"Shot {shot_num} not found")


def _get_caption_text(episode, shot_num, shot_info):
    caption_path = FRAMES_DIR / episode / f"s{shot_num:02d}_caption.txt"
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return shot_info.get("caption", "")


def _find_selected_frame(episode_dir, shot_prefix):
    matches = list(episode_dir.glob(f"{shot_prefix}_*_selected.jpg"))
    return matches[0] if matches else None


def _rebuild_dataset_shot(episode, shot_info):
    """Rebuild a single shot's dataset files."""
    shot_num = shot_info["shot"]
    shot_prefix = f"s{shot_num:02d}"
    ds_img = DATASET_DIR / f"{episode}_{shot_prefix}.jpg"
    ds_txt = DATASET_DIR / f"{episode}_{shot_prefix}.txt"

    if shot_info.get("excluded", False):
        ds_img.unlink(missing_ok=True)
        ds_txt.unlink(missing_ok=True)
        return

    episode_dir = FRAMES_DIR / episode
    selected_path = _find_selected_frame(episode_dir, shot_prefix)
    if not selected_path:
        return

    caption = _get_caption_text(episode, shot_num, shot_info)
    spy = shot_info.get("spy", "None")
    appearance_suffix = shot_info.get("appearance_suffix")
    nonstandard = shot_info.get("nonstandard_outfit", False)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_path, ds_img)
    ds_txt.write_text(
        build_caption_text(caption, spy, appearance_suffix=appearance_suffix,
                           nonstandard_outfit=nonstandard),
        encoding="utf-8",
    )


# ── Review Progress ──────────────────────────────────────────────────────────────

def load_review_progress():
    if REVIEW_PROGRESS_PATH.exists():
        data = json.loads(REVIEW_PROGRESS_PATH.read_text(encoding="utf-8"))
        return set(data.get("reviewed_episodes", []))
    return set()


def save_review_progress(reviewed):
    REVIEW_PROGRESS_PATH.write_text(
        json.dumps({"reviewed_episodes": sorted(reviewed)}, indent=2),
        encoding="utf-8",
    )


# ── Pydantic Models ─────────────────────────────────────────────────────────────

class CaptionUpdate(BaseModel):
    caption: str

class ViewUpdate(BaseModel):
    view: str

class SpyUpdate(BaseModel):
    spy1: str | None = None
    spy2: str | None = None

class HatUpdate(BaseModel):
    white_hat_off: bool = False
    black_hat_off: bool = False

class NonstandardUpdate(BaseModel):
    nonstandard_outfit: bool

class SuffixUpdate(BaseModel):
    appearance_suffix: str

class ReviewUpdate(BaseModel):
    reviewed: bool


# ── FastAPI App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Frame Reviewer Web")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return index_path.read_text(encoding="utf-8")


# -- Episodes ------------------------------------------------------------------

@app.get("/api/episodes")
async def list_episodes():
    """List all episodes with shot counts and review status."""
    reviewed = load_review_progress()
    episodes = []
    if FRAMES_DIR.exists():
        for ep_dir in sorted(FRAMES_DIR.iterdir()):
            if not ep_dir.is_dir():
                continue
            manifest_path = ep_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shots = manifest.get("shots", [])
            total = len(shots)
            excluded = sum(1 for s in shots if s.get("excluded", False))
            episodes.append({
                "name": ep_dir.name,
                "total_shots": total,
                "excluded_shots": excluded,
                "reviewed": ep_dir.name in reviewed,
            })
    return episodes


@app.post("/api/episodes/{episode}/review")
async def set_review(episode: str, body: ReviewUpdate):
    """Toggle reviewed status for an episode."""
    reviewed = load_review_progress()
    if body.reviewed:
        reviewed.add(episode)
    else:
        reviewed.discard(episode)
    save_review_progress(reviewed)
    return {"ok": True}


# -- Shots (Gallery) ----------------------------------------------------------

@app.get("/api/shots/{episode}")
async def list_shots(episode: str):
    """List shots for a single episode (gallery data)."""
    manifest = _read_manifest(episode)
    shots = []
    for shot_info in manifest["shots"]:
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        caption = _get_caption_text(episode, shot_num, shot_info)

        # Find selected frame URL
        selected = _find_selected_frame(FRAMES_DIR / episode, shot_prefix)
        selected_url = f"/frames/{episode}/{selected.name}" if selected else None

        shots.append({
            "shot": shot_num,
            "caption": caption,
            "excluded": shot_info.get("excluded", False),
            "spy": shot_info.get("spy", "None"),
            "view": shot_info.get("view", ""),
            "spy1": shot_info.get("spy1"),
            "spy2": shot_info.get("spy2"),
            "selected_frame": selected_url,
        })
    return shots


# -- Shot Detail ---------------------------------------------------------------

@app.get("/api/shot/{episode}/{shot_num}")
async def get_shot(episode: str, shot_num: int):
    """Full detail for one shot including candidate frame URLs."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)
    shot_prefix = f"s{shot_num:02d}"
    episode_dir = FRAMES_DIR / episode

    caption = _get_caption_text(episode, shot_num, shot_info)

    # Find all candidate frames
    candidates = sorted(episode_dir.glob(f"{shot_prefix}_*.jpg"))
    candidate_urls = []
    selected_idx = -1
    for i, c in enumerate(candidates):
        candidate_urls.append({
            "url": f"/frames/{episode}/{c.name}",
            "name": c.name,
            "is_selected": "_selected" in c.name,
        })
        if "_selected" in c.name:
            selected_idx = i

    return {
        "shot": shot_num,
        "episode": episode,
        "caption": caption,
        "excluded": shot_info.get("excluded", False),
        "spy": shot_info.get("spy", "None"),
        "spy1": shot_info.get("spy1"),
        "spy2": shot_info.get("spy2"),
        "view": shot_info.get("view", ""),
        "white_hat_off": shot_info.get("white_hat_off", False),
        "black_hat_off": shot_info.get("black_hat_off", False),
        "nonstandard_outfit": shot_info.get("nonstandard_outfit", False),
        "appearance_suffix": shot_info.get("appearance_suffix", ""),
        "candidates": candidate_urls,
        "selected_idx": selected_idx,
    }


@app.post("/api/shot/{episode}/{shot_num}/caption")
async def update_caption(episode: str, shot_num: int, body: CaptionUpdate):
    """Save edited caption text."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)
    shot_prefix = f"s{shot_num:02d}"

    caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
    caption_path.write_text(body.caption, encoding="utf-8")

    shot_info["caption"] = body.caption
    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True}


@app.post("/api/shot/{episode}/{shot_num}/view")
async def update_view(episode: str, shot_num: int, body: ViewUpdate):
    """Update the view/framing tag (replaces first comma-separated segment)."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)
    shot_prefix = f"s{shot_num:02d}"

    caption = _get_caption_text(episode, shot_num, shot_info)
    new_view = body.view.strip()

    # Replace first comma-separated segment
    comma_idx = caption.find(",")
    if comma_idx > 0:
        new_caption = new_view + caption[comma_idx:]
    else:
        new_caption = new_view

    caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
    caption_path.write_text(new_caption, encoding="utf-8")

    shot_info["caption"] = new_caption
    shot_info["view"] = new_view
    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True, "caption": new_caption}


@app.post("/api/shot/{episode}/{shot_num}/spy")
async def update_spy(episode: str, shot_num: int, body: SpyUpdate):
    """Update spy1/spy2 assignments, adjusting caption text accordingly."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)
    shot_prefix = f"s{shot_num:02d}"

    caption = _get_caption_text(episode, shot_num, shot_info)
    old_spy1 = shot_info.get("spy1")
    old_spy2 = shot_info.get("spy2")
    new_spy1 = body.spy1
    new_spy2 = body.spy2

    # Handle spy1 changes
    if new_spy1 != old_spy1:
        caption = _apply_spy_change(caption, "spy1", old_spy1, new_spy1,
                                    old_spy2, new_spy2)
        # Auto-swap: if new_spy1 matches old_spy2
        if new_spy1 and old_spy2 and new_spy1 == old_spy2:
            new_spy2 = old_spy1  # swap

    # Handle spy2 changes
    if new_spy2 != old_spy2:
        caption = _apply_spy_change(caption, "spy2", old_spy2, new_spy2,
                                    new_spy1, None)
        # Auto-swap: if new_spy2 matches new_spy1
        if new_spy2 and new_spy1 and new_spy2 == new_spy1:
            new_spy1 = old_spy2

    shot_info["spy1"] = new_spy1
    shot_info["spy2"] = new_spy2
    shot_info["spy"] = derive_spy_type(new_spy1, new_spy2)
    shot_info["caption"] = caption

    # Recompute appearance suffix
    derived = shot_info["spy"]
    nonstandard = shot_info.get("nonstandard_outfit", False)
    shot_info["appearance_suffix"] = compute_appearance_suffix(
        derived,
        white_hat_off=shot_info.get("white_hat_off", False),
        black_hat_off=shot_info.get("black_hat_off", False),
        nonstandard_outfit=nonstandard,
    )

    # Reset hat checkboxes if spy type no longer applicable
    if derived in ("White", "None"):
        shot_info["black_hat_off"] = False
    if derived in ("Black", "None"):
        shot_info["white_hat_off"] = False

    caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
    caption_path.write_text(caption, encoding="utf-8")
    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {
        "ok": True,
        "caption": caption,
        "spy": shot_info["spy"],
        "spy1": shot_info["spy1"],
        "spy2": shot_info["spy2"],
        "white_hat_off": shot_info["white_hat_off"],
        "black_hat_off": shot_info["black_hat_off"],
        "appearance_suffix": shot_info["appearance_suffix"],
    }


def _apply_spy_change(caption, which, old_val, new_val, other_val, other_new):
    """Apply a single spy field change to caption text."""
    if old_val and new_val and old_val != new_val:
        # Check for auto-swap scenario
        if new_val == other_val:
            caption = swap_spies_in_caption(caption, old_val, new_val)
        else:
            caption = replace_spy_in_caption(caption, old_val, new_val)
    elif old_val and not new_val:
        caption = strip_spy_names(caption)
    return caption


@app.post("/api/shot/{episode}/{shot_num}/hats")
async def update_hats(episode: str, shot_num: int, body: HatUpdate):
    """Update hat-off flags and recompute appearance suffix."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)

    shot_info["white_hat_off"] = body.white_hat_off
    shot_info["black_hat_off"] = body.black_hat_off

    spy = shot_info.get("spy", "None")
    nonstandard = shot_info.get("nonstandard_outfit", False)
    shot_info["appearance_suffix"] = compute_appearance_suffix(
        spy,
        white_hat_off=body.white_hat_off,
        black_hat_off=body.black_hat_off,
        nonstandard_outfit=nonstandard,
    )

    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True, "appearance_suffix": shot_info["appearance_suffix"]}


@app.post("/api/shot/{episode}/{shot_num}/nonstandard")
async def update_nonstandard(episode: str, shot_num: int, body: NonstandardUpdate):
    """Toggle non-standard outfit flag and recompute suffix."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)

    shot_info["nonstandard_outfit"] = body.nonstandard_outfit

    spy = shot_info.get("spy", "None")
    shot_info["appearance_suffix"] = compute_appearance_suffix(
        spy,
        white_hat_off=shot_info.get("white_hat_off", False),
        black_hat_off=shot_info.get("black_hat_off", False),
        nonstandard_outfit=body.nonstandard_outfit,
    )

    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True, "appearance_suffix": shot_info["appearance_suffix"]}


@app.post("/api/shot/{episode}/{shot_num}/suffix")
async def update_suffix(episode: str, shot_num: int, body: SuffixUpdate):
    """Update the appearance suffix directly."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)

    shot_info["appearance_suffix"] = body.appearance_suffix
    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True}


@app.post("/api/shot/{episode}/{shot_num}/exclude")
async def toggle_exclude(episode: str, shot_num: int):
    """Toggle excluded status for a shot."""
    manifest = _read_manifest(episode)
    shot_info = _find_shot(manifest, shot_num)

    shot_info["excluded"] = not shot_info.get("excluded", False)
    _write_manifest(episode, manifest)
    _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True, "excluded": shot_info["excluded"]}


@app.post("/api/shot/{episode}/{shot_num}/select/{candidate_name}")
async def select_candidate(episode: str, shot_num: int, candidate_name: str):
    """Change selected frame for a shot."""
    episode_dir = FRAMES_DIR / episode
    shot_prefix = f"s{shot_num:02d}"

    # Validate candidate exists
    candidate_path = episode_dir / candidate_name
    if not candidate_path.exists():
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Remove _selected from current selection
    for old in episode_dir.glob(f"{shot_prefix}_*_selected.jpg"):
        new_name = old.name.replace("_selected", "")
        old.rename(episode_dir / new_name)

    # Add _selected to the chosen candidate
    clean_name = candidate_name.replace("_selected", "")
    source = episode_dir / clean_name
    if source.exists():
        new_name = clean_name.replace(".jpg", "_selected.jpg")
        source.rename(episode_dir / new_name)

        # Update manifest
        manifest = _read_manifest(episode)
        shot_info = _find_shot(manifest, shot_num)
        # Extract the candidate index from filename: s01_03.jpg -> "03"
        idx_part = clean_name.replace(f"{shot_prefix}_", "").replace(".jpg", "")
        shot_info["selected"] = idx_part
        _write_manifest(episode, manifest)
        _rebuild_dataset_shot(episode, shot_info)

    return {"ok": True}


# -- Static frame serving -----------------------------------------------------

@app.get("/frames/{episode}/{filename}")
async def serve_frame(episode: str, filename: str):
    """Serve frame images from the frames/ directory."""
    file_path = FRAMES_DIR / episode / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(file_path, media_type="image/jpeg")


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting Frame Reviewer Web on http://0.0.0.0:8420")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Frames dir:   {FRAMES_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8420)
