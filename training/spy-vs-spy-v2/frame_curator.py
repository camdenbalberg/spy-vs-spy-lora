#!/usr/bin/env python3
"""
Frame Curator + Auto-Captioning App
Curate animation frames with voice-guided captioning via Claude Vision API.

Usage:
    python frame_curator.py           # Normal curation mode
    python frame_curator.py --test    # Test mode (refine prompt before bulk captioning)

Controls:
    Left/Right   Navigate frames
    Space        Keep frame + start recording voice note
    Enter        Stop recording, transcribe, send to API
    Escape       Cancel recording (deletes copied frame)
    D / Delete   Skip frame
    Backspace    Go back
    E            Edit caption
    R            Re-record voice note for last kept frame
    Q            Quit
"""

import os
import sys
import io
import wave
import shutil
import base64
import threading
import argparse
import tkinter as tk
from pathlib import Path
from collections import OrderedDict

import numpy as np
import sounddevice as sd
from PIL import Image, ImageTk

try:
    import anthropic
except ImportError:
    print("Missing: pip install anthropic")
    sys.exit(1)

try:
    import speech_recognition as sr
except ImportError:
    print("Missing: pip install speechrecognition")
    sys.exit(1)

# === Configuration ===
FRAMES_DIR = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\frames"
SELECTED_DIR = r"C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\selected"
PROMPT_FILE = "spyvspy_caption_prompt.md"
CACHE_AHEAD = 5
CACHE_BEHIND = 5
CACHE_MAX = 20
SAMPLE_RATE = 16000

# Caption wrappers
PREFIX = "spyvspy, "
SUFFIX_BOTH = (
    ", both wearing fedora hats and trenchcoats with long pointed beak noses"
    " and black sclera eyes, black and white ink comic art, bold outlines,"
    " high contrast, slapstick cartoon style"
)
SUFFIX_SINGLE = (
    ", wearing a fedora hat and trenchcoat with a long pointed beak nose"
    " and black sclera eyes, black and white ink comic art, bold outlines,"
    " high contrast, slapstick cartoon style"
)
SUFFIX_NONE = (
    ", black and white ink comic art, bold outlines, high contrast,"
    " slapstick cartoon style"
)


def wrap_caption(response_text):
    t = response_text.lower()
    has_black = "black spy" in t
    has_white = "white spy" in t
    has_grey = "grey spy" in t or "gray spy" in t
    spy_count = sum([has_black, has_white, has_grey])
    if spy_count >= 2:
        suffix = SUFFIX_BOTH
    elif spy_count == 1:
        suffix = SUFFIX_SINGLE
    else:
        suffix = SUFFIX_NONE
    return PREFIX + response_text.strip().rstrip(".") + suffix


# === Audio Recorder ===

class AudioRecorder:
    """Records audio from default mic using sounddevice, transcribes with speech_recognition."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self._recording = False
        self._frames = []
        self._stream = None

    def start(self):
        self._frames = []
        self._recording = True
        # Find the real mic (skip virtual audio devices like SteelSeries Sonar)
        device = self._find_mic()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=self._callback, device=device,
        )
        self._stream.start()

    @staticmethod
    def _find_mic():
        """Find a real microphone, skipping virtual audio devices."""
        devices = sd.query_devices()
        # Prefer hardware mics over virtual ones
        skip = ("sonar", "virtual", "steam", "stereo mix", "line in")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                name = d["name"].lower()
                if not any(s in name for s in skip):
                    return i
        return None  # fallback to default

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def stop(self):
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def cancel(self):
        self.stop()
        self._frames = []

    def transcribe(self):
        """Convert recorded audio to text. Returns transcription string."""
        if not self._frames:
            return ""
        audio_np = np.concatenate(self._frames)
        # Write WAV to memory buffer
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_np.tobytes())
        buf.seek(0)
        # Feed to speech_recognition
        with sr.AudioFile(buf) as source:
            audio_data = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "[inaudible]"
        except sr.RequestError as e:
            return f"[transcription error: {e}]"


# === Image Cache ===

class ImageCache:
    def __init__(self, frames_dir, frame_list, screen_w, screen_h):
        self.frames_dir = frames_dir
        self.frame_list = frame_list
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._pil = OrderedDict()
        self._photo = {}
        self._lock = threading.Lock()
        self._loading = set()

    def _load_scaled(self, index):
        path = os.path.join(self.frames_dir, self.frame_list[index])
        img = Image.open(path).convert("RGB")
        max_w, max_h = self.screen_w, self.screen_h
        w, h = img.size
        ratio = min(max_w / w, max_h / h)
        if ratio != 1.0:
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    def get_photo(self, index):
        if index in self._photo:
            return self._photo[index]
        with self._lock:
            pil = self._pil.get(index)
        if pil is None:
            pil = self._load_scaled(index)
            with self._lock:
                self._pil[index] = pil
                self._evict()
        photo = ImageTk.PhotoImage(pil)
        self._photo[index] = photo
        for k in list(self._photo):
            if abs(k - index) > CACHE_AHEAD + 3:
                del self._photo[k]
        return photo

    def preload_around(self, center, total):
        lo = max(0, center - CACHE_BEHIND)
        hi = min(total, center + CACHE_AHEAD + 1)
        for i in range(lo, hi):
            with self._lock:
                if i in self._pil or i in self._loading:
                    continue
                self._loading.add(i)
            threading.Thread(target=self._preload_one, args=(i,), daemon=True).start()

    def _preload_one(self, index):
        try:
            img = self._load_scaled(index)
            with self._lock:
                self._pil[index] = img
                self._evict()
        finally:
            with self._lock:
                self._loading.discard(index)

    def _evict(self):
        while len(self._pil) > CACHE_MAX:
            self._pil.popitem(last=False)


# === App States ===
BROWSING = "browsing"
RECORDING = "recording"
EDITING = "editing"


class FrameCuratorApp:
    def __init__(self, root, test_mode=False):
        self.root = root
        self.test_mode = test_mode
        self.state = BROWSING

        # Directories
        self.frames_dir = FRAMES_DIR
        self.selected_dir = SELECTED_DIR
        os.makedirs(self.selected_dir, exist_ok=True)

        # Frame list — natural sort so s1_9 < s1_10 < s1_100
        import re
        def _nat_key(name):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]

        self.frames = sorted(
            (f for f in os.listdir(self.frames_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))),
            key=_nat_key,
        )
        if not self.frames:
            print(f"No images in {self.frames_dir}")
            sys.exit(1)

        # State
        self.current_index = 0
        self.kept_frames = set()
        self.captions = {}
        self.voice_notes = {}
        self.pending_captions = 0
        self.last_kept = None
        self.recording_filename = None

        # Audio
        self.recorder = AudioRecorder()

        # Test mode
        self.test_selections = []
        self.test_ratings = {}
        self.in_review = False
        self.review_index = 0

        # Preload timer
        self._preload_timer = None

        # Load existing
        self._load_existing()

        # API — resolve key from env or Windows registry
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                    api_key, _ = winreg.QueryValueEx(key, "ANTHROPIC_API_KEY")
            except Exception:
                pass
        if not api_key:
            print("ANTHROPIC_API_KEY not found. Set it as a system environment variable.")
            sys.exit(1)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.caption_prompt = self._load_prompt()

        # Build UI
        self._build_ui()
        self._bind_keys()

        # Cache created after window is mapped
        self.cache = None
        self.root.after(200, self._init_cache_and_show)

    def _load_existing(self):
        if not os.path.isdir(self.selected_dir):
            return
        latest_time = 0
        for f in os.listdir(self.selected_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                self.kept_frames.add(f)
                stem = Path(f).stem
                txt = Path(self.selected_dir) / (stem + ".txt")
                voice = Path(self.selected_dir) / (stem + "_voice.txt")
                if txt.exists():
                    self.captions[f] = txt.read_text(encoding="utf-8")
                if voice.exists():
                    self.voice_notes[f] = voice.read_text(encoding="utf-8")
                # Track most recently kept frame
                mtime = os.path.getmtime(os.path.join(self.selected_dir, f))
                if mtime > latest_time:
                    latest_time = mtime
                    self.last_kept = f

    def _load_prompt(self):
        candidates = [
            Path(__file__).resolve().parent / PROMPT_FILE,
            Path(self.frames_dir).parent / PROMPT_FILE,
        ]
        for p in candidates:
            if p.exists():
                return p.read_text(encoding="utf-8")
        print(f"Warning: {PROMPT_FILE} not found")
        return "Describe this animation frame in detail."

    def _init_cache_and_show(self):
        self.root.update_idletasks()
        sw = self.root.winfo_width()
        sh = self.root.winfo_height()
        if sw < 100:
            sw = self.root.winfo_screenwidth()
        if sh < 100:
            sh = self.root.winfo_screenheight()
        self.cache = ImageCache(self.frames_dir, self.frames, sw, sh)
        self._show_current()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.configure(bg="black")
        self.root.geometry("1600x900")
        self.root.state("zoomed")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.img_id = None

        # All overlays start at 0,0 — repositioned dynamically in _layout_overlays
        self.txt_counter = self.canvas.create_text(
            0, 0, anchor="nw", fill="white",
            font=("Consolas", 16, "bold"), text="",
        )
        self.txt_filename = self.canvas.create_text(
            0, 0, anchor="n", fill="#aaaaaa",
            font=("Consolas", 13), text="",
        )
        self.txt_kept = self.canvas.create_text(
            0, 0, anchor="ne", fill="#00ff00",
            font=("Consolas", 14, "bold"), text="",
        )
        self.txt_pending = self.canvas.create_text(
            0, 0, anchor="ne", fill="#ffaa00",
            font=("Consolas", 12), text="",
        )
        self.txt_indicator = self.canvas.create_text(
            0, 0, anchor="n", fill="#00ff00",
            font=("Consolas", 13, "bold"), text="",
        )
        self.txt_recording = self.canvas.create_text(
            0, 0, anchor="center", fill="#ff0000",
            font=("Consolas", 24, "bold"), text="",
        )
        self.txt_voice = self.canvas.create_text(
            0, 0, anchor="s", fill="#88ccff",
            font=("Consolas", 11), text="",
        )
        self.txt_caption = self.canvas.create_text(
            0, 0, anchor="s", fill="#ffff00",
            font=("Consolas", 11), text="",
        )
        self.txt_mode = self.canvas.create_text(
            0, 0, anchor="sw", fill="#ff00ff",
            font=("Consolas", 11, "bold"), text="",
        )

        self.canvas.bind("<Configure>", self._on_canvas_resize)

        if self.test_mode:
            self.canvas.itemconfig(
                self.txt_mode,
                text="TEST MODE  |  Space=select+record  Enter=stop recording  Q=quit",
            )

        self._layout_overlays()

        # Edit overlay
        self.edit_frame = tk.Frame(self.root, bg="#1a1a2e")
        self.edit_label = tk.Label(
            self.edit_frame,
            text="  Edit caption  |  Ctrl+Enter = save  |  Escape = cancel  ",
            bg="#1a1a2e", fg="#aaaaaa", font=("Consolas", 10), anchor="w",
        )
        self.edit_label.pack(fill=tk.X, padx=10, pady=(5, 0))
        self.edit_text = tk.Text(
            self.edit_frame, wrap=tk.WORD, bg="#222244", fg="white",
            font=("Consolas", 12), height=4, insertbackground="white",
            relief=tk.FLAT, padx=8, pady=5,
        )
        self.edit_text.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _layout_overlays(self):
        sw = self.canvas.winfo_width()
        sh = self.canvas.winfo_height()
        if sw < 100:
            sw = 1600
        if sh < 100:
            sh = 900

        self.canvas.coords(self.txt_counter, 20, 15)
        self.canvas.coords(self.txt_filename, sw // 2, 15)
        self.canvas.coords(self.txt_kept, sw - 20, 15)
        self.canvas.coords(self.txt_pending, sw - 20, 45)
        self.canvas.coords(self.txt_indicator, sw // 2, 45)
        self.canvas.coords(self.txt_recording, sw // 2, sh // 2 + 80)
        self.canvas.coords(self.txt_voice, sw // 2, sh - 85)
        self.canvas.coords(self.txt_caption, sw // 2, sh - 15)
        self.canvas.coords(self.txt_mode, 20, sh - 15)

        self.canvas.itemconfig(self.txt_voice, width=sw - 80)
        self.canvas.itemconfig(self.txt_caption, width=sw - 80)

    def _on_canvas_resize(self, event=None):
        self._layout_overlays()
        # Rebuild cache with new dimensions if significant size change
        sw = self.canvas.winfo_width()
        sh = self.canvas.winfo_height()
        if self.cache and sw > 100 and sh > 100:
            self.cache.screen_w = sw
            self.cache.screen_h = sh

    def _bind_keys(self):
        self.root.bind("<Right>", self._on_right)
        self.root.bind("<Left>", self._on_left)
        self.root.bind("<Control-Right>", self._on_right_10)
        self.root.bind("<Control-Left>", self._on_left_10)
        self.root.bind("<Control-Shift-Right>", self._on_right_50)
        self.root.bind("<Control-Shift-Left>", self._on_left_50)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Delete>", self._on_skip)
        self.root.bind("<d>", self._on_skip)
        self.root.bind("<BackSpace>", self._on_left)
        self.root.bind("<e>", self._on_edit)
        self.root.bind("<r>", self._on_rerecord)
        self.root.bind("<l>", self._on_goto_last_kept)
        self.root.bind("<q>", self._on_quit)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Return>", self._on_enter)

    def _all_overlays(self):
        return (
            self.txt_counter, self.txt_filename, self.txt_kept,
            self.txt_pending, self.txt_indicator, self.txt_recording,
            self.txt_voice, self.txt_caption, self.txt_mode,
        )

    def _raise_overlays(self):
        for item in self._all_overlays():
            self.canvas.tag_raise(item)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _show_current(self):
        if self.in_review:
            self._show_review_frame()
        else:
            self._show_frame(self.current_index)

    def _show_frame(self, index):
        if index < 0 or index >= len(self.frames):
            return
        self.current_index = index
        fname = self.frames[index]

        photo = self.cache.get_photo(index)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        if self.img_id:
            self.canvas.delete(self.img_id)
        self.img_id = self.canvas.create_image(
            sw // 2, sh // 2, anchor="center", image=photo,
        )
        self._raise_overlays()

        # Counters
        self.canvas.itemconfig(
            self.txt_counter, text=f"{index + 1:,} / {len(self.frames):,}",
        )
        self.canvas.itemconfig(self.txt_filename, text=fname)

        if self.test_mode and not self.in_review:
            self.canvas.itemconfig(
                self.txt_kept, text=f"Test selected: {len(self.test_selections)}",
            )
        else:
            self.canvas.itemconfig(
                self.txt_kept, text=f"Kept: {len(self.kept_frames)}",
            )

        self._update_pending()

        # Indicator
        if self.test_mode and fname in [s for s in self.test_selections]:
            self.canvas.itemconfig(self.txt_indicator, text="[SELECTED]", fill="#ff00ff")
        elif fname in self.kept_frames:
            self.canvas.itemconfig(self.txt_indicator, text="[KEPT]", fill="#00ff00")
        else:
            self.canvas.itemconfig(self.txt_indicator, text="")

        # Voice note
        if fname in self.voice_notes:
            self.canvas.itemconfig(self.txt_voice, text=f"Voice: {self.voice_notes[fname]}")
        else:
            self.canvas.itemconfig(self.txt_voice, text="")

        # Caption
        if fname in self.captions:
            self.canvas.itemconfig(self.txt_caption, text=self.captions[fname])
        else:
            self.canvas.itemconfig(self.txt_caption, text="")

        # Clear recording indicator if not recording
        if self.state != RECORDING:
            self.canvas.itemconfig(self.txt_recording, text="")

        self._schedule_preload()

    def _update_pending(self):
        if self.pending_captions > 0:
            self.canvas.itemconfig(
                self.txt_pending,
                text=f"Captioning... ({self.pending_captions} pending)",
            )
        else:
            self.canvas.itemconfig(self.txt_pending, text="")

    def _schedule_preload(self):
        if self._preload_timer:
            self.root.after_cancel(self._preload_timer)
        self._preload_timer = self.root.after(
            150, lambda: self.cache.preload_around(self.current_index, len(self.frames)),
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_right(self, event=None):
        if self.state != BROWSING:
            return
        if self.in_review:
            self._review_next()
        elif self.current_index < len(self.frames) - 1:
            self._show_frame(self.current_index + 1)

    def _on_left(self, event=None):
        if self.state != BROWSING:
            return
        if self.in_review:
            self._review_prev()
        elif self.current_index > 0:
            self._show_frame(self.current_index - 1)

    def _on_right_10(self, event=None):
        if self.state != BROWSING or self.in_review:
            return
        self._show_frame(min(self.current_index + 10, len(self.frames) - 1))

    def _on_left_10(self, event=None):
        if self.state != BROWSING or self.in_review:
            return
        self._show_frame(max(self.current_index - 10, 0))

    def _on_right_50(self, event=None):
        if self.state != BROWSING or self.in_review:
            return
        self._show_frame(min(self.current_index + 50, len(self.frames) - 1))

    def _on_left_50(self, event=None):
        if self.state != BROWSING or self.in_review:
            return
        self._show_frame(max(self.current_index - 50, 0))

    def _on_goto_last_kept(self, event=None):
        if self.state != BROWSING or not self.last_kept:
            return
        try:
            idx = self.frames.index(self.last_kept)
            self._show_frame(idx)
        except ValueError:
            pass

    def _on_skip(self, event=None):
        if self.state != BROWSING:
            return
        self._on_right()

    # ------------------------------------------------------------------
    # Space: Keep + Record
    # ------------------------------------------------------------------

    def _on_space(self, event=None):
        if self.state == RECORDING:
            return
        if self.state == EDITING:
            return

        fname = self.frames[self.current_index]

        # Already kept — just advance
        if fname in self.kept_frames and not self.test_mode:
            self._on_right()
            return

        # Copy frame to selected/
        src = os.path.join(self.frames_dir, fname)
        dst = os.path.join(self.selected_dir, fname)
        shutil.copy2(src, dst)

        if self.test_mode and not self.in_review:
            if fname not in self.test_selections:
                self.test_selections.append(fname)

        self.kept_frames.add(fname)
        self.recording_filename = fname

        # Start recording
        self.state = RECORDING
        self.recorder.start()

        # Update UI
        self.canvas.itemconfig(self.txt_indicator, text="[KEPT]", fill="#00ff00")
        self.canvas.itemconfig(
            self.txt_recording, text="\U0001f534  RECORDING...  (Enter to stop, Esc to cancel)",
        )
        self.canvas.itemconfig(
            self.txt_kept,
            text=(f"Test selected: {len(self.test_selections)}"
                  if self.test_mode and not self.in_review
                  else f"Kept: {len(self.kept_frames)}"),
        )

    # ------------------------------------------------------------------
    # Enter: Stop recording + transcribe + caption
    # ------------------------------------------------------------------

    def _on_enter(self, event=None):
        if self.state == RECORDING:
            self._stop_recording_and_caption()
            return

        # Test mode: start review
        if self.test_mode and not self.in_review and self.test_selections:
            self._start_review()

    def _stop_recording_and_caption(self):
        fname = self.recording_filename
        self.recorder.stop()

        # Grab the recorded audio frames immediately so recorder is free for reuse
        audio_frames = list(self.recorder._frames)
        self.recorder._frames = []

        self.last_kept = fname
        self.state = BROWSING
        self.canvas.itemconfig(self.txt_recording, text="")

        # Immediately advance to next frame — zero delay
        if self.current_index < len(self.frames) - 1:
            self._show_frame(self.current_index + 1)

        # Bump pending counter (transcription + captioning combined)
        self.pending_captions += 1
        self._update_pending()

        # Entire pipeline runs in background: transcribe → save voice → send to API
        threading.Thread(
            target=self._transcribe_and_caption_worker,
            args=(fname, audio_frames),
            daemon=True,
        ).start()

    def _transcribe_and_caption_worker(self, fname, audio_frames):
        """Background worker: transcribe audio, save voice note, send caption request."""
        # Transcribe
        if audio_frames:
            audio_np = np.concatenate(audio_frames)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())
            buf.seek(0)
            recognizer = sr.Recognizer()
            with sr.AudioFile(buf) as source:
                audio_data = recognizer.record(source)
            try:
                voice_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                voice_text = "[inaudible]"
            except sr.RequestError as e:
                voice_text = f"[transcription error: {e}]"
        else:
            voice_text = ""

        # Store and save voice note
        self.voice_notes[fname] = voice_text
        stem = Path(fname).stem
        voice_path = os.path.join(self.selected_dir, stem + "_voice.txt")
        with open(voice_path, "w", encoding="utf-8") as f:
            f.write(voice_text)

        # Update voice text on UI if this frame is still displayed
        self.root.after(0, self._update_voice_if_visible, fname, voice_text)

        # Read image and send to API (reuses pending counter already incremented)
        img_path = os.path.join(self.selected_dir, fname)
        with open(img_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = Path(fname).suffix.lower()
        media = "image/png" if ext == ".png" else "image/jpeg"

        try:
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f'The user describes this frame as: "{voice_text}"\n\n'
                        "Using this description as guidance and the image as "
                        "visual reference, write the detailed caption."
                    ),
                },
            ]
            msg = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.3,
                system=self.caption_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            caption = wrap_caption(msg.content[0].text)
            self.root.after(0, self._caption_done, fname, caption)
        except Exception as e:
            self.root.after(0, self._caption_error, fname, str(e))

    def _update_voice_if_visible(self, fname, voice_text):
        """Show voice transcription on UI if the user is still looking at this frame."""
        cur = self.frames[self.current_index]
        if cur == fname:
            self.canvas.itemconfig(self.txt_voice, text=f"Voice: {voice_text}")

    # ------------------------------------------------------------------
    # Escape: Cancel recording
    # ------------------------------------------------------------------

    def _on_escape(self, event=None):
        if self.state == RECORDING:
            self.recorder.cancel()
            self.state = BROWSING
            fname = self.recording_filename

            # Delete copied frame from selected/
            dst = os.path.join(self.selected_dir, fname)
            if os.path.exists(dst):
                os.remove(dst)
            self.kept_frames.discard(fname)
            if self.test_mode and fname in self.test_selections:
                self.test_selections.remove(fname)

            self.recording_filename = None
            self.canvas.itemconfig(self.txt_recording, text="")
            self._show_frame(self.current_index)
            return

        if self.state == EDITING:
            self._cancel_edit()
            return

        if self.in_review:
            self._exit_review()
            return

        # Default: exit app
        self.root.destroy()

    # ------------------------------------------------------------------
    # R: Re-record voice note
    # ------------------------------------------------------------------

    def _on_rerecord(self, event=None):
        if self.state != BROWSING:
            return

        # Re-record for current frame if kept, or last kept
        fname = self.frames[self.current_index]
        if fname not in self.kept_frames:
            fname = self.last_kept
        if not fname or fname not in self.kept_frames:
            return

        # Navigate to this frame if needed
        if fname in self.frames:
            idx = self.frames.index(fname)
            self._show_frame(idx)

        self.recording_filename = fname
        self.state = RECORDING
        self.recorder.start()

        self.canvas.itemconfig(
            self.txt_recording,
            text="\U0001f534  RE-RECORDING...  (Enter to stop, Esc to cancel)",
        )

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def _caption_done(self, filename, caption):
        self.captions[filename] = caption
        self.pending_captions -= 1
        self._update_pending()

        # Save .txt (training caption)
        stem = Path(filename).stem
        txt = os.path.join(self.selected_dir, stem + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(caption)

        # Update display if visible
        cur = (self.test_selections[self.review_index]
               if self.in_review
               else self.frames[self.current_index])
        if cur == filename:
            self.canvas.itemconfig(self.txt_caption, text=caption)

    def _caption_error(self, filename, error):
        self.pending_captions -= 1
        self._update_pending()
        err = f"[ERROR: {error}]"
        self.captions[filename] = err
        if self.frames[self.current_index] == filename:
            self.canvas.itemconfig(self.txt_caption, text=err)

    # ------------------------------------------------------------------
    # Edit mode
    # ------------------------------------------------------------------

    def _on_edit(self, event=None):
        if self.state != BROWSING:
            return

        if self.in_review:
            target = self.test_selections[self.review_index]
        elif self.frames[self.current_index] in self.captions:
            target = self.frames[self.current_index]
        elif self.last_kept and self.last_kept in self.captions:
            target = self.last_kept
        else:
            return

        self.state = EDITING
        self.editing_filename = target

        self.edit_text.delete("1.0", tk.END)
        self.edit_text.insert("1.0", self.captions.get(target, ""))
        self.edit_frame.place(relx=0, rely=0.80, relwidth=1, relheight=0.20)
        self.edit_text.focus_set()

        self.edit_text.bind("<Control-Return>", self._save_edit)
        self.edit_text.bind("<Escape>", self._cancel_edit)

    def _save_edit(self, event=None):
        new = self.edit_text.get("1.0", tk.END).strip()
        fname = self.editing_filename

        self.captions[fname] = new
        stem = Path(fname).stem
        txt = os.path.join(self.selected_dir, stem + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(new)

        self._cancel_edit()

        cur = (self.test_selections[self.review_index]
               if self.in_review
               else self.frames[self.current_index])
        if cur == fname:
            self.canvas.itemconfig(self.txt_caption, text=new)

        return "break"

    def _cancel_edit(self, event=None):
        self.state = BROWSING
        self.editing_filename = None
        self.edit_frame.place_forget()
        self.canvas.focus_set()

    # ------------------------------------------------------------------
    # Test mode review
    # ------------------------------------------------------------------

    def _start_review(self):
        self.in_review = True
        self.review_index = 0

        self.canvas.itemconfig(
            self.txt_mode,
            text="REVIEW  |  Arrows=nav  G=good  B=bad  E=edit  Esc=exit",
        )
        self.root.bind("<g>", self._mark_good)
        self.root.bind("<b>", self._mark_bad)
        self._show_review_frame()

    def _show_review_frame(self):
        if self.review_index < 0 or self.review_index >= len(self.test_selections):
            return
        fname = self.test_selections[self.review_index]
        frame_idx = self.frames.index(fname)

        photo = self.cache.get_photo(frame_idx)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        if self.img_id:
            self.canvas.delete(self.img_id)
        self.img_id = self.canvas.create_image(
            sw // 2, sh // 2, anchor="center", image=photo,
        )
        self._raise_overlays()

        self.canvas.itemconfig(
            self.txt_counter,
            text=f"Review {self.review_index + 1} / {len(self.test_selections)}",
        )
        self.canvas.itemconfig(self.txt_filename, text=fname)

        good = sum(1 for v in self.test_ratings.values() if v is True)
        bad = sum(1 for v in self.test_ratings.values() if v is False)
        self.canvas.itemconfig(self.txt_kept, text=f"Good: {good}  Bad: {bad}")

        rating = self.test_ratings.get(fname)
        if rating is True:
            self.canvas.itemconfig(self.txt_indicator, text="[GOOD]", fill="#00ff00")
        elif rating is False:
            self.canvas.itemconfig(self.txt_indicator, text="[NEEDS WORK]", fill="#ff4444")
        else:
            self.canvas.itemconfig(self.txt_indicator, text="[UNRATED]", fill="#888888")

        # Voice
        voice = self.voice_notes.get(fname, "")
        self.canvas.itemconfig(
            self.txt_voice, text=f"Voice: {voice}" if voice else "",
        )
        # Caption
        cap = self.captions.get(fname, "Waiting for caption...")
        self.canvas.itemconfig(self.txt_caption, text=cap)

    def _review_next(self):
        if self.review_index < len(self.test_selections) - 1:
            self.review_index += 1
            self._show_review_frame()
        else:
            self._show_summary()

    def _review_prev(self):
        if self.review_index > 0:
            self.review_index -= 1
            self._show_review_frame()

    def _mark_good(self, event=None):
        if not self.in_review or self.state != BROWSING:
            return
        fname = self.test_selections[self.review_index]
        self.test_ratings[fname] = True
        self.canvas.itemconfig(self.txt_indicator, text="[GOOD]", fill="#00ff00")
        self.root.after(200, self._review_next)

    def _mark_bad(self, event=None):
        if not self.in_review or self.state != BROWSING:
            return
        fname = self.test_selections[self.review_index]
        self.test_ratings[fname] = False
        self.canvas.itemconfig(self.txt_indicator, text="[NEEDS WORK]", fill="#ff4444")

    def _show_summary(self):
        good = sum(1 for v in self.test_ratings.values() if v is True)
        bad = sum(1 for v in self.test_ratings.values() if v is False)
        unrated = len(self.test_selections) - good - bad

        summary = (
            f"TEST SUMMARY\n\n"
            f"Total: {len(self.test_selections)}\n"
            f"Good: {good}    Needs work: {bad}    Unrated: {unrated}\n\n"
            f"Edit spyvspy_caption_prompt.md and re-run --test to iterate\n"
            f"Escape = back to review    Q = quit"
        )

        if self.img_id:
            self.canvas.delete(self.img_id)
            self.img_id = None
        self.canvas.itemconfig(self.txt_counter, text="Summary")
        self.canvas.itemconfig(self.txt_filename, text="")
        self.canvas.itemconfig(self.txt_indicator, text="")
        self.canvas.itemconfig(self.txt_voice, text="")
        self.canvas.itemconfig(self.txt_caption, text=summary)

    def _exit_review(self):
        self.in_review = False
        self.root.unbind("<g>")
        self.root.unbind("<b>")
        self.canvas.itemconfig(
            self.txt_mode,
            text="TEST MODE  |  Space=select+record  Enter=stop recording  Q=quit",
        )
        self._show_frame(self.current_index)

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _on_quit(self, event=None):
        if self.state == RECORDING:
            self.recorder.cancel()
        if self.state != EDITING:
            self.root.destroy()


def main():
    global FRAMES_DIR, SELECTED_DIR, PROMPT_FILE

    parser = argparse.ArgumentParser(description="Frame Curator for LoRA training")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--frames", default=FRAMES_DIR, help="Frames directory")
    parser.add_argument("--output", default=SELECTED_DIR, help="Output directory")
    parser.add_argument("--prompt", default=None, help="Caption prompt file")
    args = parser.parse_args()

    FRAMES_DIR = args.frames
    SELECTED_DIR = args.output
    if args.prompt:
        PROMPT_FILE = args.prompt

    if not os.path.isdir(FRAMES_DIR):
        print(f"Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)

    root = tk.Tk()
    root.title("Frame Curator")
    FrameCuratorApp(root, test_mode=args.test)
    root.mainloop()


if __name__ == "__main__":
    main()
