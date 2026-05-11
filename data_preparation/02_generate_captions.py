"""
Step 2 — Generate surgical captions from triplet annotations.

Each triplet (instrument, verb, target) is turned into a natural-language
caption describing the surgical action.  The script supports two backends:

  "template"  (default, no API key needed)
    Rule-based sentence templates.  Fast and reproducible; produces shorter,
    more formulaic captions but requires no external service.

  "openai"
    Calls the OpenAI Chat API (GPT-4o by default).  Produces richer,
    context-aware descriptions matching the style of HuluMed captions used in
    the paper.  Requires OPENAI_API_KEY in the environment.

  "qwen"
    Calls the DashScope API (Qwen2.5-72B-Instruct by default).
    Requires DASHSCOPE_API_KEY in the environment.

Input
-----
  clips_metadata.json — output of 01_extract_action_clips.py

Output
------
  clips_captions.json — same entries, each with an added "caption" field

Usage
-----
  # Template (fast, no API key)
  python 02_generate_captions.py --metadata data/cholec80_action/clips_metadata.json

  # OpenAI GPT-4o
  export OPENAI_API_KEY=sk-...
  python 02_generate_captions.py \\
      --metadata  data/cholec80_action/clips_metadata.json \\
      --backend   openai  --model gpt-4o \\
      --max_workers 16

  # Qwen (DashScope)
  export DASHSCOPE_API_KEY=sk-...
  python 02_generate_captions.py \\
      --metadata  data/cholec80_action/clips_metadata.json \\
      --backend   qwen  --model qwen2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# ── vocabulary helpers ────────────────────────────────────────────────────────

_INSTRUMENT_DESCRIPTIONS = {
    "grasper":   "a laparoscopic grasper",
    "bipolar":   "bipolar forceps",
    "hook":      "an electrosurgical hook",
    "scissors":  "laparoscopic scissors",
    "clipper":   "a clip applier",
    "irrigator": "an irrigation-suction device",
    "null":      "an instrument",
}

_VERB_PHRASES = {
    "grasp":      "grasps",
    "retract":    "retracts",
    "dissect":    "dissects",
    "coagulate":  "coagulates",
    "clip":       "applies a clip to",
    "cut":        "cuts",
    "aspirate":   "aspirates fluid from",
    "irrigate":   "irrigates",
    "pack":       "packs",
    "null":       "interacts with",
}

_TARGET_PHRASES = {
    "gallbladder":           "the gallbladder",
    "cystic_plate":          "the cystic plate",
    "cystic_duct":           "the cystic duct",
    "cystic_pedicle":        "the cystic pedicle",
    "cystic_artery":         "the cystic artery",
    "gallbladder_peritoneum":"the gallbladder peritoneum",
    "gallbladder_wall":      "the gallbladder wall",
    "liver":                 "the liver",
    "adhesion":              "an adhesion",
    "omentum":               "the omentum",
    "peritoneum":            "the peritoneum",
    "gut":                   "the gut",
    "specimen_bag":          "a specimen bag",
    "fluid":                 "the accumulated fluid",
    "null":                  "the target tissue",
}

_BACKGROUND_SENTENCES = [
    "The liver and surrounding fatty tissue are visible in the background.",
    "The liver forms a prominent dark-red backdrop, with yellowish fatty tissue nearby.",
    "Bright laparoscopic lighting illuminates the abdominal cavity clearly.",
    "The peritoneum and surrounding structures remain stable throughout.",
    "The operative field is well-exposed with clear visibility.",
]

_CONTEXT_SENTENCES = {
    ("grasper", "grasp"):    "The grasper's jaws maintain a secure hold throughout the maneuver.",
    ("hook",    "dissect"):  "The electrosurgical hook carefully separates tissue planes.",
    ("hook",    "coagulate"):"The hook tip applies controlled electrosurgical energy to the tissue.",
    ("clipper", "clip"):     "Titanium clips are deployed with precision at the target structure.",
    ("scissors","cut"):      "The scissors deliver a clean, controlled incision.",
    ("irrigator","aspirate"):"The suction cannula efficiently removes fluid from the operative field.",
    ("irrigator","irrigate"):"Saline is gently irrigated to clear the operative field.",
    ("bipolar", "coagulate"):"Bipolar energy is applied carefully to achieve hemostasis.",
}


def template_caption(triplets: List[Tuple[str, str, str]]) -> str:
    """Generate a structured caption from a list of triplets using templates."""
    sentences = []

    for inst, verb, tgt in triplets:
        inst_desc = _INSTRUMENT_DESCRIPTIONS.get(inst, f"a {inst}")
        verb_phrase = _VERB_PHRASES.get(verb, verb)
        tgt_phrase  = _TARGET_PHRASES.get(tgt, tgt.replace('_', ' '))
        sentences.append(
            f"The surgical procedure involves {inst_desc} that {verb_phrase} {tgt_phrase}."
        )

    # Add a context sentence if we recognise the (instrument, verb) pair
    for inst, verb, tgt in triplets:
        ctx = _CONTEXT_SENTENCES.get((inst, verb))
        if ctx:
            sentences.append(ctx)
            break

    # Add one fixed background sentence
    sentences.append(random.choice(_BACKGROUND_SENTENCES))

    # Closing observation
    instruments_used = list({inst for inst, _, _ in triplets})
    if len(instruments_used) == 1:
        sentences.append(
            f"The motion is deliberate and controlled, typical of laparoscopic "
            f"cholecystectomy technique."
        )

    return " ".join(sentences)


# ── LLM backends ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a medical imaging expert describing surgical video clips for a "
    "machine-learning dataset. Given a list of instrument-verb-target triplets "
    "from a laparoscopic cholecystectomy, write a single descriptive paragraph "
    "(5–8 sentences) about the surgical action depicted. Be specific about the "
    "instrument, the motion, and the anatomical target. Mention the laparoscopic "
    "environment (lighting, tissue texture, surrounding structures). Do not use "
    "bullet points or headers. Write in present tense."
)


def _triplets_to_prompt(triplets: List[Tuple[str, str, str]]) -> str:
    lines = [f"  - {inst} / {verb} / {tgt}" for inst, verb, tgt in triplets]
    return "Triplet annotations:\n" + "\n".join(lines)


def openai_caption(
    triplets: List[Tuple[str, str, str]],
    model: str = "gpt-4o",
    max_retries: int = 5,
) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _triplets_to_prompt(triplets)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  OpenAI error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return template_caption(triplets)   # fall back to template on repeated failure


def qwen_caption(
    triplets: List[Tuple[str, str, str]],
    model: str = "qwen2.5-72b-instruct",
    max_retries: int = 5,
) -> str:
    import dashscope
    from dashscope import Generation
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    prompt = _triplets_to_prompt(triplets)
    for attempt in range(max_retries):
        try:
            resp = Generation.call(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=512,
                temperature=0.7,
                result_format="message",
            )
            return resp.output.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  Qwen error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return template_caption(triplets)


def generate_caption(
    triplets: List[Tuple[str, str, str]],
    backend: str,
    model: Optional[str],
) -> str:
    trips = [tuple(t) for t in triplets]
    if backend == "openai":
        return openai_caption(trips, model=model or "gpt-4o")
    if backend == "qwen":
        return qwen_caption(trips, model=model or "qwen2.5-72b-instruct")
    return template_caption(trips)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    meta_path = Path(args.metadata)
    with open(meta_path) as f:
        metadata = json.load(f)

    out_path = meta_path.parent / "clips_captions.json"

    # Load any existing output so we can resume
    existing: dict = {}
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for entry in json.load(f):
                existing[entry["video"]] = entry.get("caption", "")

    todo = [m for m in metadata if m["video"] not in existing or not existing[m["video"]]]
    print(f"Total clips   : {len(metadata)}")
    print(f"Already done  : {len(existing)}")
    print(f"To generate   : {len(todo)}")
    print(f"Backend       : {args.backend}" + (f" / {args.model}" if args.model else ""))

    results: dict[str, str] = dict(existing)

    def _process(entry):
        vid  = entry["video"]
        caps = generate_caption(entry["triplets"], args.backend, args.model)
        return vid, caps

    if args.max_workers > 1 and args.backend != "template":
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_process, m): m for m in todo}
            done = 0
            for fut in as_completed(futures):
                vid, cap = fut.result()
                results[vid] = cap
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)} captions generated...")
                    _save(metadata, results, out_path)
    else:
        for i, entry in enumerate(todo):
            vid, cap = _process(entry)
            results[vid] = cap
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(todo)} captions generated...")
                _save(metadata, results, out_path)

    _save(metadata, results, out_path)
    print(f"\nSaved {len(results)} captions → {out_path}")
    print("Next step: python ../scripts/prepare_cholec80_laof.py")


def _save(metadata: list, results: dict, out_path: Path) -> None:
    out = []
    for m in metadata:
        vid = m["video"]
        entry = {
            "video":      vid,
            "video_path": str(Path("clips") / vid),
            "triplets":   m["triplets"],
            "caption":    results.get(vid, ""),
        }
        out.append(entry)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate captions for surgical action clips from triplet annotations."
    )
    p.add_argument("--metadata", default="data/cholec80_action/clips_metadata.json",
                   help="Path to clips_metadata.json (output of 01_extract_action_clips.py).")
    p.add_argument("--backend", default="template",
                   choices=["template", "openai", "qwen"],
                   help="Caption generation backend (default: template).")
    p.add_argument("--model", default=None,
                   help="LLM model name (optional; uses per-backend default).")
    p.add_argument("--max_workers", type=int, default=8,
                   help="Parallel API workers for openai/qwen backends (default: 8).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate captions for clips that already have one.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
