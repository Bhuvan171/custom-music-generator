"""
Generate music from a text prompt, with an optional reference track for secondary control.

    python generate.py --prompt "a slow, moody piano ballad with brushed drums"
    python generate.py --prompt "heavy metal guitar" --reference song.mp3

TEXT IS NOW THE PRIMARY CONDITIONING SIGNAL (the text-primary redesign)
-------------------------------------------------------------------------
`--prompt` is encoded LIVE by a frozen T5-base text encoder and fed to the DiT via
cross-attention — real language understanding (T5's own pretraining), not a lookup into a fixed
vocabulary. This replaces the old retrieval-only path (src/text_tags.py, which matched free text
to 195 fixed tags and could not express anything beyond them). Tags are now an AUXILIARY signal
only: if your prompt happens to match a tag confidently, it's added to the cheap global
conditioning vector too, but a prompt that matches nothing still works fully via T5.

`--reference` (any audio file) remains available as SECONDARY control: its per-frame harmony
(chroma), energy, and texture (onset/percussiveness/brightness/noisiness) are added on top of
the text conditioning. It is not required — that was the entire point of moving to real text
conditioning — but for now, with the tag-only conditioning that predates this redesign, it
measurably helped (latent std 1.19 vs 0.64, flatness 1.06x vs 0.44x without one). Whether
text-only generation is now good enough on its own is an open, not-yet-measured question for
the NEW model — see train_dit.py's `clap_gap` metric once training exists.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.dit import MusicDiT
from src.features import extract_all, load_reference
from src.vae import WaveformVAE

_ft_ckpts = sorted(glob.glob(os.path.join(config.FT_DEC_CKPT_DIR, "dec_step*.pt")))
VAE_CKPT  = _ft_ckpts[-1] if _ft_ckpts else "checkpoints/vae/vae_step0200000.pt"
REFERENCE_DIR = "references"
AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".aiff")


def encode_prompt_t5(text, device):
    """
    Live T5 encoding of the user's prompt — same frozen model, same tokenization
    (padding="max_length", max_length=TEXT_MAX_TOKENS) as compute_text_embeddings.py uses to
    cache training captions, so train and inference embedding spaces match exactly.
    Returns (1, TEXT_MAX_TOKENS, TEXT_DIM) hidden states + (1, TEXT_MAX_TOKENS) bool mask.
    """
    from transformers import T5EncoderModel, T5Tokenizer
    if not hasattr(encode_prompt_t5, "_model"):
        encode_prompt_t5._tok   = T5Tokenizer.from_pretrained(config.T5_MODEL_NAME)
        encode_prompt_t5._model = T5EncoderModel.from_pretrained(config.T5_MODEL_NAME).to(device)
        encode_prompt_t5._model.eval()
        encode_prompt_t5._model.requires_grad_(False)
    tok, model = encode_prompt_t5._tok, encode_prompt_t5._model
    enc = tok([text], return_tensors="pt", padding="max_length", truncation=True,
              max_length=config.TEXT_MAX_TOKENS).to(device)
    with torch.no_grad():
        emb = model(**enc).last_hidden_state.float()   # (1, TEXT_MAX_TOKENS, 768)
    return emb, enc["attention_mask"].bool()


def list_references():
    if not os.path.isdir(REFERENCE_DIR):
        return []
    return sorted(f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith(AUDIO_EXT))


def resolve_reference(ref):
    """A real path wins; otherwise treat it as a filename inside references/."""
    if os.path.isfile(ref):
        return ref
    cand = os.path.join(REFERENCE_DIR, ref)
    if os.path.isfile(cand):
        return cand
    # tolerate a missing/incorrect extension: match on the stem
    stem = os.path.splitext(os.path.basename(ref))[0].lower()
    for f in list_references():
        if os.path.splitext(f)[0].lower() == stem:
            return os.path.join(REFERENCE_DIR, f)
    have = list_references()
    raise SystemExit(
        f"No such reference: {ref!r}\n" +
        (f"In {REFERENCE_DIR}/: {have}" if have else
         f"{REFERENCE_DIR}/ is empty — drop an audio file in it (see {REFERENCE_DIR}/README.md), "
         f"or pass a full path to any audio file."))


def resolve_tags(spec, vocab):
    """'genre---rock, instrument---bass' -> [idx, ...], with a useful error on a typo."""
    if not spec:
        return []
    out = []
    for raw in spec.split(","):
        t = raw.strip()
        if not t:
            continue
        if t in vocab:
            out.append(vocab[t])
            continue
        # Accept a bare name ("rock") if it is unambiguous across the three categories.
        hits = [k for k in vocab if k.split("---")[-1] == t]
        if len(hits) == 1:
            out.append(vocab[hits[0]])
            print(f"  ('{t}' -> '{hits[0]}')")
        elif len(hits) > 1:
            raise SystemExit(f"'{t}' is ambiguous: {hits}. Use the full name.")
        else:
            near = [k for k in vocab if t.lower() in k.lower()][:8]
            raise SystemExit(f"Unknown tag '{t}'." + (f" Did you mean: {near}" if near else
                             " Run --list-tags to see the vocabulary."))
    return out


def main():
    p = argparse.ArgumentParser(description="Generate music from a text prompt + a reference track.")
    p.add_argument("--prompt", default=None,
                   help="FREE TEXT, e.g. 'a slow, moody piano ballad with brushed drums'. Encoded "
                        "LIVE by a frozen T5 encoder and fed to the DiT via cross-attention -- "
                        "real language understanding, not a lookup table. Also opportunistically "
                        "matched against the 195-tag vocabulary (src/text_tags.py) for the cheap "
                        "AUXILIARY global signal; --explain shows that match without generating.")
    p.add_argument("--tags", default="", help="exact tags, ADDED to whatever --prompt gives (both "
                        "feed the same auxiliary adaLN signal), e.g. 'genre---rock,instrument---bass'")
    p.add_argument("--explain", action="store_true",
                   help="show the AUXILIARY tag match for --prompt, then exit (does not reflect "
                        "the primary T5 cross-attention path, which has no comparable "
                        "'explanation' -- it's a continuous embedding, not a lookup)")
    p.add_argument("--reference", default=None,
                   help="audio file whose harmony/texture guides the output. A bare filename is "
                        "looked up in references/ , so `--reference mysong.mp3` just works. "
                        "Strongly recommended: without it the model regresses to the mean.")
    p.add_argument("--list-references", action="store_true", help="show what is in references/")
    p.add_argument("--offset", type=float, default=0.0, help="start N seconds into the reference")
    p.add_argument("--n", type=int, default=4, help="how many variations to generate")
    p.add_argument("--cfg-scale", type=float, default=config.CFG_SCALE,
                   help="conditioning strength. 1.0 = most natural variance, 1.5 = follows the "
                        "reference harmony more closely, >2 over-drives.")
    p.add_argument("--steps", type=int, default=config.EULER_STEPS)
    p.add_argument("--seed", type=int, default=None)
    _dit_ckpts = (sorted(glob.glob(os.path.join(config.DIT_FT_CKPT_DIR, "dit_step*.pt")))
                 or sorted(glob.glob(os.path.join(config.DIT_CKPT_DIR, "dit_step*.pt"))))
    p.add_argument("--checkpoint", default=(_dit_ckpts[-1] if _dit_ckpts else None),
                   help="DiT checkpoint (default: latest in checkpoints/dit_ft/ or checkpoints/dit/)")
    p.add_argument("--vae-checkpoint", default=VAE_CKPT,
                   help="override the frozen VAE decoder, e.g. a checkpoint from "
                        "finetune_decoder.py (checkpoints/vae_decoder_ft/dec_stepXXXXXXX.pt)")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--list-tags", action="store_true")
    p.add_argument("--search", default=None, help="filter the tag list by substring")
    args = p.parse_args()

    if args.list_references:
        have = list_references()
        if have:
            print(f"{REFERENCE_DIR}/ ({len(have)} files):")
            for f in have:
                print(f"  {f}")
            print(f"\nUse:  python generate.py --prompt \"...\" --reference {have[0]}")
        else:
            print(f"{REFERENCE_DIR}/ is empty. Drop any .mp3/.wav/.flac in it, then:")
            print(f'  python generate.py --prompt "upbeat jazzy piano" --reference yourfile.mp3')
            print(f"See {REFERENCE_DIR}/README.md")
        return

    with open(config.VOCAB_PATH) as f:
        vocab = json.load(f)

    if args.list_tags or args.search:
        from collections import defaultdict
        groups = defaultdict(list)
        for t in sorted(vocab):
            if args.search and args.search.lower() not in t.lower():
                continue
            cat, name = t.split("---")
            groups[cat].append(name)
        for cat, names in groups.items():
            print(f"\n{cat} ({len(names)}):")
            for i in range(0, len(names), 6):
                print("  " + "  ".join(f"{n:<20}" for n in names[i:i + 6]))
        return

    # Tags are now AUXILIARY: --prompt's PRIMARY effect is the live T5 encoding below (feats
    # dict, cross-attention). This block only resolves the cheap secondary adaLN tag signal, and
    # --prompt / --tags are additive (both can contribute tags at once), not either/or.
    tag_idx = list(resolve_tags(args.tags, vocab)) if args.tags else []
    if args.prompt:
        from src.text_tags import TextTagMatcher
        matches = TextTagMatcher().match(args.prompt)
        print(f"Prompt: {args.prompt!r}")
        if matches:
            for t, s in matches:
                print(f"    {s:.2f}  {t}  (auxiliary tag match)")
            tag_idx += [vocab[t] for t, _ in matches if vocab[t] not in tag_idx]
        else:
            print("    (no auxiliary tag matched confidently -- fine, the primary T5 "
                  "cross-attention path does not need one)")
        if args.explain:
            return
    if not tag_idx and not args.reference and not args.prompt:
        raise SystemExit("Give --prompt, --tags or --reference. See --list-tags.")
    if args.checkpoint is None:
        raise SystemExit("No DiT checkpoint found in checkpoints/dit_ft/ or checkpoints/dit/, and "
                          "none given via --checkpoint. Train one first (python src/train_dit.py).")

    device = torch.device("cuda")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    vae = WaveformVAE().to(device)
    _missing, _unexpected = vae.load_state_dict(
        torch.load(args.vae_checkpoint, map_location=device, weights_only=False)["vae"], strict=False)
    if _missing or _unexpected:
        print(f"  Partial VAE load from {args.vae_checkpoint} (head architecture differs): "
              f"missing={_missing} unexpected={_unexpected}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    vae.eval(); vae.requires_grad_(False)

    dit = MusicDiT().to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dit.load_state_dict(ck["ema"], strict=False)   # EMA weights, as in every eval
    dit.eval()
    print(f"DiT {args.checkpoint} (step {ck['step']})   CFG {args.cfg_scale}   {args.steps} steps")

    # ---- conditioning -------------------------------------------------------------------
    # Built additively, exactly like src/dataset.py's __getitem__: text (PRIMARY) and reference
    # (SECONDARY) each contribute their own keys to ONE feats dict, so either can be used alone
    # or both together. MusicDiT.forward() reads only the keys that are present.
    feats = {}

    if args.prompt:
        text_emb, text_mask = encode_prompt_t5(args.prompt, device)   # (1,T,768), (1,T) bool
        feats["text_emb"]   = text_emb.repeat(args.n, 1, 1)
        feats["text_mask"]  = text_mask.repeat(args.n, 1)
        feats["text_valid"] = torch.ones(args.n, device=device)
        print(f"Text conditioning: live T5 encoding of the prompt (primary signal)")

    if args.reference:
        ref_path = resolve_reference(args.reference)
        audio, sr = load_reference(ref_path, offset=args.offset)
        f = extract_all(audio, sr)          # the SAME recipe the training features were built with
        # The model consumes STANDARDIZED features; the dataset does this during training, so a
        # reference conditioned on raw values would be far out of distribution.
        with open(config.FEATURE_STATS_PATH) as fh:
            st = json.load(fh)
        def std_(x, k):
            m = torch.tensor(st[f"{k}_mean"]).view(-1, 1)
            s = torch.tensor(st[f"{k}_std"]).view(-1, 1)
            return (torch.from_numpy(x).float() - m) / s
        feats.update({
            "tempo":   torch.tensor([f["tempo"]]).repeat(args.n),
            "key":     torch.tensor([f["key"]]).repeat(args.n),
            "mode":    torch.tensor([f["mode"]]).repeat(args.n),
            "chroma":  std_(f["chroma"],  "chroma").unsqueeze(0).repeat(args.n, 1, 1),
            "texture": std_(f["texture"], "texture").unsqueeze(0).repeat(args.n, 1, 1),
            "valid":   torch.ones(args.n),
        })
        key_name = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][f["key"]]
        print(f"Reference (secondary): {ref_path}  ({args.offset:.0f}s+)  "
              f"tempo {f['tempo']:.0f} BPM, key {key_name} {'major' if f['mode'] else 'minor'}")
    else:
        print("No --reference given — text-only generation. This is the intended default for "
              "the text-primary redesign; whether it's good enough on its own is a live "
              "question until the new cross-attention model is actually trained and measured "
              "(train_dit.py's clap_gap is the metric that will answer it).")

    feats = {k: v.to(device) for k, v in feats.items()} if feats else None
    names = [k for k, v in sorted(vocab.items(), key=lambda kv: kv[1]) if v in set(tag_idx)]
    print(f"Auxiliary tags: {names if names else '(none)'}")

    # ---- generate ----------------------------------------------------------------------
    with open(config.LATENT_STATS_PATH) as fh:
        ls = json.load(fh)
    lmean = torch.tensor(ls["mean"], device=device).view(1, -1, 1)
    lstd  = torch.tensor(ls["std"],  device=device).view(1, -1, 1)

    with torch.no_grad():
        z = dit.sample([tag_idx] * args.n, steps=args.steps, cfg_scale=args.cfg_scale,
                       device=device, feats=feats)
        audio_out = vae.decode(z.float() * lstd + lmean).float()   # denormalize, then decode

    out_dir = args.out or os.path.join("samples", "generated",
                                       datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.n):
        x = audio_out[i, 0].cpu().numpy()
        peak = float(np.abs(x).max())
        # sf.write defaults to PCM_16 and HARD-CLIPS outside [-1,1]; the VAE decoder overshoots
        # full scale even from ground-truth latents (measured peak 2.00), and clipping
        # manufactures broadband distortion that sounds exactly like the artifact we chase.
        sf.write(os.path.join(out_dir, f"gen{i}.wav"),
                 x / peak if peak > 1.0 else x, config.SAMPLE_RATE)
        print(f"  gen{i}.wav" + (f"   [peak {peak:.2f} -> normalized]" if peak > 1.0 else ""))
    if args.reference:
        sf.write(os.path.join(out_dir, "reference.wav"), audio, config.SAMPLE_RATE)
        print("  reference.wav  (what the harmony was taken from)")
    print(f"\n{args.n} clips -> {out_dir}")


if __name__ == "__main__":
    main()
