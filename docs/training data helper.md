# FRAME CURATION + AUTO-CAPTIONING APP — BUILD SPEC

## Purpose
Desktop app for curating animation frames and auto-captioning them via Claude Vision API for SDXL LoRA training data.

## File Paths

- **Frame source:** `C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\frames\`
  - Contains PNGs named like `s1_0001.png`, `s2_0001.png`, etc.
  - Expect 10,000-50,000 frames total.

- **Output:** `C:\Users\Camden\Documents\Coding Projects\LoRA training\training\spy-vs-spy\selected\`
  - Kept frames get copied here with TWO matching text files:
    - `s1_0423.txt` — Final wrapped caption from Claude API (matches image name exactly — required for kohya_ss training)
    - `s1_0423_voice.txt` — Raw voice transcription (your verbal description, saved for reference/debugging)
  - The frame itself keeps its original name: `s1_0423.png`
  - Example directory contents:
    ```
    s1_0423.png
    s1_0423.txt          ← training caption
    s1_0423_voice.txt    ← your voice note
    s2_0100.png
    s2_0100.txt
    s2_0100_voice.txt
    ```

- **Caption prompt file:** `spyvspy_caption_prompt.md` (same directory as the app, or passed as argument)
  - The app reads this file and uses its entire contents as the system prompt for the Claude API call.
  - This keeps the prompt editable without touching app code.

## UI Requirements

- Fullscreen window showing the current frame scaled to fit
- Current filename and frame count displayed as overlay (e.g., "1,342 / 48,291")
- Count of how many frames have been kept so far
- When a frame is selected, show the generated caption at the bottom of the screen
- Show a small "Captioning..." indicator while waiting for the API response
- Show a small counter of pending caption requests (API calls are async)

## Controls

- **Left/Right arrow keys:** Navigate frames (must be fast, no lag — preload next few frames)
- **Spacebar:** Keep current frame — copy to `selected/` folder, immediately start recording audio for a voice note
- **Enter:** Stop recording, transcribe voice note, send image + voice note to Claude API for captioning, save caption as .txt when response returns
- **Escape (while recording):** Cancel the keep — stop recording, delete the copied frame from `selected/`, go back to browsing
- **Delete or D key:** Skip frame, advance to next
- **Backspace:** Go back to previous frame
- **E key:** Edit the last generated caption — open a text input with the current caption text, let me modify it, Enter to save the edit
- **R key:** Re-run captioning on the last selected frame (re-record voice note + re-send to API)
- **Q:** Quit app

## Voice Note Recording

When spacebar is pressed:
1. Show a prominent "🔴 Recording..." indicator on screen
2. Start capturing audio from the default microphone
3. User speaks a brief description (5-15 words, e.g., "white spy lighting bomb in briefcase at bus stop")
4. User hits **Enter** to stop recording
5. Transcribe audio using `speech_recognition` library (Google Speech API or Windows SAPI)
6. Display the transcription on screen briefly
7. Save transcription as `[filename]_voice.txt` in `selected/`
8. Send transcription + image to Claude API

The voice note is a quick anchor to tell Claude what's happening in the frame. It doesn't need to be detailed — Claude Vision + the prompt handle the detail. The voice note prevents misidentification.

## Claude API Captioning

When a frame is selected with Spacebar and voice note is recorded:

1. Copy the frame to `selected/`
2. Record and transcribe voice note
3. Read the prompt from `spyvspy_caption_prompt.md`
4. Send to Claude API with:
   - **Model:** `claude-sonnet-4-20250514`
   - **Max tokens:** 300
   - **Temperature:** 0.3
   - **System prompt:** Contents of `spyvspy_caption_prompt.md`
   - **User message:** The image as base64 with media type `image/png`, followed by the text:
     ```
     The user describes this frame as: "{transcribed_voice_note}"
     
     Using this description as guidance and the image as visual reference, write the detailed caption.
     ```
4. Take Claude's response and wrap it with prefix/suffix. The suffix changes based on how many spies are in the scene — check if the response contains both "black spy" and "white spy", or just one:

   **Both spies:**
   ```
   spyvspy, [response], both wearing fedora hats and trenchcoats with long pointed beak noses and black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
   ```

   **Single spy:**
   ```
   spyvspy, [response], wearing a fedora hat and trenchcoat with a long pointed beak nose and black sclera eyes, black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
   ```

   **No spy (response contains neither "black spy" nor "white spy" nor "grey spy"):**
   ```
   spyvspy, [response], black and white ink comic art, bold outlines, high contrast, slapstick cartoon style
   ```

   **Note:** Grey spy counts as a spy for the wrapper. If grey spy + one other spy, use the "both" wrapper. If grey spy alone, use the "single" wrapper.
5. Save the voice transcription as `[filename]_voice.txt` in `selected/`
6. Save the wrapped caption as `[filename].txt` in `selected/` (matches image name for training)
7. Display the full caption on screen

## API Configuration

- Use the `anthropic` Python library
- API key: Read from environment variable `ANTHROPIC_API_KEY`
- API calls must be **async/non-blocking** — the UI must not freeze while waiting for a response. I should be able to keep browsing and selecting frames while captions generate in the background.

## Performance Requirements

- Must handle 50,000+ frames without loading them all into memory
- Preload ~5 frames ahead and behind for instant navigation
- Arrow key holding should rapidly scroll through frames (like holding an arrow key in a file browser)
- API calls must not block the UI

## Tech Stack

- Python, tkinter or PyQt5
- `anthropic` library for API calls
- `speech_recognition` library for voice-to-text
- Pillow for image loading
- `pip install anthropic pillow speechrecognition pyaudio` (and PyQt5 if using that)

## Platform

- Windows 11, Python 3.13

## Testing Phase

Before full curation mode, the app should have a **test mode** (launched with a `--test` flag or a button in the UI):

1. **Select test frames:** User picks 10-20 frames covering different scenarios (solo spy, duo, celebration, close-up, wide shot, disguise, explosion aftermath, non-spy characters, tiny distant spy, no spy visible).
2. **Run captioning:** Send all selected test frames through the API.
3. **Side-by-side review:** Display each frame with its generated caption below. Arrow keys to navigate between test results.
4. **Edit/approve:** Let user edit captions inline. Mark each as "good" or "needs work".
5. **Summary:** At the end, show stats — how many were good vs needed editing, common issues.
6. **Purpose:** This lets the user refine `spyvspy_caption_prompt.md` before committing to 1,000+ API calls. The user edits the prompt file, re-runs test mode, and compares until quality is acceptable.