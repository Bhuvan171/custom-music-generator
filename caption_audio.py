"""
Caption every track by LISTENING to the audio, via an audio-language model (Qwen2-Audio).

THIS IS THE LOAD-BEARING DECISION OF THE TEXT-PRIMARY REDESIGN. An LLM paraphrasing the existing
195 tags ("an energetic rock track with electric guitar") adds ZERO information beyond the tags
themselves — a text encoder trained on such captions inherits the exact ~30-40 bit/track ceiling
the tags-only DiT already hit (this was measured directly: LP-MusicCaps-style tag paraphrasing is
a known dead end for exactly this reason). An audio-language model that processes the actual
waveform can describe instrumentation, texture, and mood that never appeared in any tag — that is
new information, and it's the only reason this redesign can outperform the retrieval-based text
path (src/text_tags.py) it replaces.

Two backends:
  --backend vllm         (default) batched offline inference, the throughput path for 55,609
                          tracks. Verify Qwen2-Audio is supported by your installed vLLM version
                          BEFORE running this on the full set (`python -c "import vllm"` and check
                          vLLM's supported-models list — audio multimodal support is newer than
                          text/vision and version-sensitive). Uses AutoProcessor.apply_chat_template
                          to build the exact prompt string, which is the version-robust way to get
                          the model's special audio tokens right (hand-written token strings break
                          across checkpoint/processor versions).
  --backend transformers slower (no batching across the whole set in one call), but has no version
                          risk — a safe fallback if the vLLM audio path doesn't work out of the box.

Usage:
  python caption_audio.py --start 12000 --workers-note "single GPU, no multiprocessing needed:
      vLLM batches internally"
  python caption_audio.py --start 10000 --limit 2000 --out-dir data/captions_val   # held-out
  python caption_audio.py --backend transformers --limit 50   # quick spot-check before committing
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

PROMPT = (
    "Listen to this music and describe it in one dense paragraph: genre, instrumentation, "
    "mood, tempo feel, and texture (e.g. smooth, gritty, sparse, lush, percussive). "
    "Be specific and concrete, not generic."
)
AUDIO_SR = 16_000   # Qwen2-Audio's expected input rate -- NOT this project's 22050Hz native rate.


def _track_id(path):
    return f"track_{int(Path(path).stem.split('_')[0]):07d}"


def load_tags_hint(track_id, tsv_path):
    """The 'blended' decision: append the curated tags as a hint alongside what the model hears,
    rather than relying on either alone."""
    if not hasattr(load_tags_hint, "_cache"):
        cache = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] == "TRACK_ID":
                    continue
                cache[parts[0]] = [t.strip() for t in parts[5:] if t.strip()]
        load_tags_hint._cache = cache
    return load_tags_hint._cache.get(track_id, [])


def load_audio_16k(path):
    import librosa
    audio, _ = librosa.load(path, sr=AUDIO_SR, mono=True, duration=config.CLIP_DURATION)
    return audio.astype(np.float32)


def already_captioned(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path) as f:
            d = json.load(f)
        return d.get("version") == config.CAPTION_VERSION and bool(d.get("caption"))
    except Exception:
        return False


def write_caption(out_path, caption, tags_hint):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": config.CAPTION_VERSION, "caption": caption, "tags_hint": tags_hint}, f)
    os.replace(tmp, out_path)   # atomic: a killed job can leave a stray .tmp, never a half-written .json


# ── vLLM backend (throughput path) ──────────────────────────────────────────────────────────

def run_vllm(files, out_dir, tsv_path, batch_size):
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.QWEN_AUDIO_MODEL)
    llm = LLM(model=config.QWEN_AUDIO_MODEL, limit_mm_per_prompt={"audio": 1}, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.3, max_tokens=config.CAPTION_MAX_NEW_TOKENS)

    pending, meta = [], []
    ok = skip = fail = 0

    def flush():
        nonlocal ok, fail
        if not pending:
            return
        outputs = llm.generate(pending, sampling_params=sampling)
        for out, (out_path, tags_hint) in zip(outputs, meta):
            try:
                text = out.outputs[0].text.strip()
                if not text:
                    raise ValueError("empty caption")
                write_caption(out_path, text, tags_hint)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  fail {out_path}: {e}")
        pending.clear(); meta.clear()

    for i, path in enumerate(files):
        tid = _track_id(path)
        out_path = os.path.join(out_dir, f"{tid}.json")
        if already_captioned(out_path):
            skip += 1
            continue
        tags_hint = load_tags_hint(tid, tsv_path)
        hint_str = f" (curated tags: {', '.join(tags_hint)})" if tags_hint else ""
        try:
            audio = load_audio_16k(path)
        except Exception as e:
            fail += 1
            print(f"  fail {tid} (load): {e}")
            continue

        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": PROMPT + hint_str},
        ]}]
        prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        pending.append({"prompt": prompt_text, "multi_modal_data": {"audio": (audio, AUDIO_SR)}})
        meta.append((out_path, tags_hint))

        if len(pending) >= batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(files)}  ok={ok} skip={skip} fail={fail}", flush=True)
    flush()
    return ok, skip, fail


# ── transformers backend (safe fallback, one track at a time) ──────────────────────────────

def run_transformers(files, out_dir, tsv_path):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(config.QWEN_AUDIO_MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.QWEN_AUDIO_MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    ok = skip = fail = 0
    for i, path in enumerate(files):
        tid = _track_id(path)
        out_path = os.path.join(out_dir, f"{tid}.json")
        if already_captioned(out_path):
            skip += 1
            continue
        tags_hint = load_tags_hint(tid, tsv_path)
        hint_str = f" (curated tags: {', '.join(tags_hint)})" if tags_hint else ""
        try:
            audio = load_audio_16k(path)
            conv = [{"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": PROMPT + hint_str},
            ]}]
            text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text_prompt, audios=[audio], sampling_rate=AUDIO_SR,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=config.CAPTION_MAX_NEW_TOKENS)
            gen = gen[:, inputs["input_ids"].shape[1]:]
            caption = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if not caption:
                raise ValueError("empty caption")
            write_caption(out_path, caption, tags_hint)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  fail {tid}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}  ok={ok} skip={skip} fail={fail}", flush=True)
    return ok, skip, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=config.CAPTIONS_DIR,
                    help="use config.CAPTIONS_VAL_DIR for the held-out slice -- MUST stay a "
                         "separate directory, same reason data/features_val/ is separate: "
                         "training set = latents INTERSECT captions.")
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--batch-size", type=int, default=32, help="vLLM backend only")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(config.STEMS_DIR, "**", "*.flac"), recursive=True))
    files = files[args.start:]
    if args.limit:
        files = files[:args.limit]
    print(f"Captioning {len(files)} tracks [{args.start}:{args.start + len(files)}] "
          f"-> {args.out_dir}/  (backend={args.backend}, model={config.QWEN_AUDIO_MODEL})")

    if args.backend == "vllm":
        ok, skip, fail = run_vllm(files, args.out_dir, config.TSV_PATH, args.batch_size)
    else:
        ok, skip, fail = run_transformers(files, args.out_dir, config.TSV_PATH)

    print(f"\nDone. ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
