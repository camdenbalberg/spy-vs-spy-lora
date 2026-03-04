MULTI-PASS CAPTIONING SYSTEM
============================

We send each video through 5 focused Gemini prompts (in the same conversation),
then merge the results with Claude API into a final LoRA training caption.

  Pass 1 — Scene & Layout (who, where, action, framing, direction)
  Pass 2 — Expressions & Body Language (emotion, posture, visual cues)
  Pass 3 — Outfits & Appearance (hat on/off, standard vs disguise)
  Pass 4 — Objects & Props (held items, interacted items, placed items)
  Pass 5 — Background & Environment (setting, location, features)

All passes output CSV with matching timestamps. Claude merges them into a single
comma-separated caption + metadata (spy1, spy2, hat states, outfit flag, view).

spy1 = the first/primary spy in the scene (e.g., "black")
spy2 = the second spy if present (e.g., "white"), or null if only one spy