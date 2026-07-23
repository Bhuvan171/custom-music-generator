"""
Frozen T5-base encodes every caption to its TOKEN-EMBEDDING SEQUENCE (not a pooled vector) --
the DiT's cross-attention needs the full sequence to attend over, exactly like Stable Audio
Open's T5 conditioning. Captions are fixed text, so this is computed once and cached; at
INFERENCE the user's live prompt is encoded on the fly with this same frozen encoder (same
model, so train/inference embedding spaces match exactly -- this is what fixed the earlier
train/inference LATENT mismatch discussion, applied here to text).

T5 is FROZEN. This is the "no pretrained weights" rule broken deliberately (see config.py's
T5_MODEL_NAME comment) -- training a text encoder from scratch on ~55k captions cannot
generalize to phrasing outside that exact set, which directly conflicts with "natural language
as the primary interface."

Usage:
  python compute_text_embeddings.py                                    # data/captions -> data/text_emb
  python compute_text_embeddings.py --captions-dir data/captions_val --out-dir data/text_emb_val
"""

import argparse
import json
import os
import sys
from glob import glob

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def already_embedded(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        d = torch.load(out_path, weights_only=True)
        return (d.get("caption_version") == config.CAPTION_VERSION
               and d.get("emb_version") == config.TEXT_EMB_VERSION)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", default=config.CAPTIONS_DIR)
    ap.add_argument("--out-dir", default=config.TEXT_EMB_DIR)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from transformers import T5EncoderModel, T5Tokenizer

    device = torch.device("cuda")
    tok = T5Tokenizer.from_pretrained(config.T5_MODEL_NAME)
    t5 = T5EncoderModel.from_pretrained(config.T5_MODEL_NAME).to(device)
    t5.eval()
    t5.requires_grad_(False)
    print(f"T5 params: {sum(p.numel() for p in t5.parameters()) / 1e6:.1f}M (frozen)")

    os.makedirs(args.out_dir, exist_ok=True)
    caption_files = sorted(glob(os.path.join(args.captions_dir, "*.json")))
    print(f"Encoding {len(caption_files)} captions from {args.captions_dir}/ -> {args.out_dir}/")

    ok = skip = fail = 0
    batch_paths, batch_texts = [], []

    def flush():
        nonlocal ok, fail
        if not batch_paths:
            return
        enc = tok(batch_texts, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=config.TEXT_MAX_TOKENS)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = t5(**enc).last_hidden_state.half().cpu()   # (B, TEXT_MAX_TOKENS, 768)
        mask = enc["attention_mask"].bool().cpu()             # (B, TEXT_MAX_TOKENS)
        for i, out_path in enumerate(batch_paths):
            try:
                tmp = out_path + ".tmp"
                torch.save({
                    "caption_version": config.CAPTION_VERSION,
                    "emb_version": config.TEXT_EMB_VERSION,
                    "emb": out[i].clone(),      # .clone() -- see cache_latents.py's bloat bug:
                    "mask": mask[i].clone(),    # a batch-slice view serializes the WHOLE batch.
                }, tmp)
                os.replace(tmp, out_path)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  fail {out_path}: {e}")
        batch_paths.clear(); batch_texts.clear()

    for i, cpath in enumerate(caption_files):
        track_id = os.path.basename(cpath)[:-5]   # strip ".json"
        out_path = os.path.join(args.out_dir, f"{track_id}.pt")
        if already_embedded(out_path):
            skip += 1
            continue
        try:
            with open(cpath) as f:
                d = json.load(f)
            caption = d["caption"]
            if d.get("tags_hint"):
                pass   # tags already folded into the caption text by caption_audio.py's prompt
        except Exception as e:
            fail += 1
            print(f"  fail {track_id} (read): {e}")
            continue
        batch_paths.append(out_path)
        batch_texts.append(caption)
        if len(batch_paths) >= args.batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(caption_files)}  ok={ok} skip={skip} fail={fail}", flush=True)
    flush()

    print(f"\nDone. ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
