import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

import config
from config import (CHUNK_SAMPLES, LATENT_STATS_PATH, FEATURES_DIR,
                    DIT_CHROMA_IN, DIT_TEXTURE_IN, VAE_LATENT_LEN)


def build_vocab(tsv_path, vocab_path):
    tags = set()
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == "TRACK_ID":
                continue
            for t in parts[5:]:
                t = t.strip()
                if t:
                    tags.add(t)
    vocab = {tag: i for i, tag in enumerate(sorted(tags))}
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Built vocab: {len(vocab)} tags → {vocab_path}")
    return vocab


def _track_id_from_path(path):
    stem = Path(path).stem          # "214", "214_0003", or "track_0000214" (cached latent)
    parts = stem.split("_")
    numeric = parts[1] if parts[0] == "track" else parts[0]
    return f"track_{int(numeric):07d}"


class JamendoDataset(Dataset):
    def __init__(self, data_dir, tsv_path, vocab_path, normalize_latents=False,
                 deterministic_crop=False, load_features=False, features_dir=None,
                 load_text=False, text_dir=None):
        # deterministic_crop: take the FIRST CHUNK_SAMPLES instead of a random window.
        # Required whenever cached latents must stay frame-aligned with separately
        # extracted features (extract_features.py) — a random crop would silently
        # misalign chroma against the latent it is supposed to describe.
        #
        # features_dir: which feature set to read. Defaults to the training set
        # (config.FEATURES_DIR); the held-out evaluator passes config.FEATURES_VAL_DIR.
        #
        # load_text / text_dir: same pattern as features_dir, for cached T5 text embeddings
        # (compute_text_embeddings.py). Defaults to config.TEXT_EMB_DIR (training); the held-out
        # evaluator passes config.TEXT_EMB_VAL_DIR. MUST stay separate for the same reason
        # features_val does: train = latents INTERSECT captions INTERSECT text_emb.
        self.deterministic_crop = deterministic_crop
        self.load_features      = load_features
        self.normalize_latents  = normalize_latents
        self.features_dir       = features_dir or FEATURES_DIR
        self.load_text          = load_text
        self.text_dir           = text_dir or config.TEXT_EMB_DIR

        # Channel standardization for chroma/texture. Applied HERE rather than in the model so
        # that train_dit / sample_dit / eval_dit all get it automatically, exactly like the
        # existing latent normalization above. Val features are normalized with TRAIN stats.
        self.feat_stats = None
        if load_features and config.DIT_STANDARDIZE_FEATS:
            if not os.path.exists(config.FEATURE_STATS_PATH):
                raise FileNotFoundError(
                    f"{config.FEATURE_STATS_PATH} not found but DIT_STANDARDIZE_FEATS=True.\n"
                    f"Run: python compute_feature_stats.py"
                )
            with open(config.FEATURE_STATS_PATH) as f:
                st = json.load(f)
            self.feat_stats = {
                k: (torch.tensor(st[f"{k}_mean"]).view(-1, 1),
                    torch.tensor(st[f"{k}_std"]).view(-1, 1))
                for k in ("chroma", "texture")
            }
        if normalize_latents:
            with open(LATENT_STATS_PATH) as f:
                stats = json.load(f)
            self.latent_mean = torch.tensor(stats["mean"]).view(-1, 1)  # (32, 1)
            self.latent_std  = torch.tensor(stats["std"]).view(-1, 1)   # (32, 1)

        # Build or load vocab
        if not os.path.exists(vocab_path):
            self.vocab = build_vocab(tsv_path, vocab_path)
        else:
            with open(vocab_path) as f:
                self.vocab = json.load(f)

        # Parse TSV: track_id → list of tag indices
        self.track_tags = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] == "TRACK_ID":
                    continue
                track_id = parts[0]
                indices = []
                for t in parts[5:]:
                    t = t.strip()
                    if t in self.vocab:
                        indices.append(self.vocab[t])
                self.track_tags[track_id] = indices

        # Glob all files
        flacs = glob(os.path.join(data_dir, "**", "*.flac"), recursive=True)
        pts   = glob(os.path.join(data_dir, "**", "*.pt"),   recursive=True)
        self.files = sorted(flacs + pts)
        print(f"Dataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        track_id = _track_id_from_path(path)
        tag_indices = self.track_tags.get(track_id, [])

        if path.endswith(".flac"):
            audio, _ = sf.read(path)                              # (T,) float32
            if len(audio) >= CHUNK_SAMPLES:
                start = 0 if self.deterministic_crop else random.randint(0, len(audio) - CHUNK_SAMPLES)
                audio = audio[start : start + CHUNK_SAMPLES]
            else:
                audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, 660480)

        elif path.endswith(".pt"):
            tensor = torch.load(path, weights_only=True).float()  # (32, 645)
            if self.normalize_latents:
                tensor = (tensor - self.latent_mean) / self.latent_std

        else:
            raise ValueError(f"Unsupported file type: {path}")

        if not (self.load_features or self.load_text):
            return tensor, tag_indices

        # Built additively so either conditioning source can be toggled independently; both
        # write into the SAME dict (collate_fn is already generic over whatever keys are
        # present, so no change needed there as long as every item in a batch has the same keys,
        # which holds since load_features/load_text are dataset-level, not per-item, flags).
        feats = {}
        if self.load_text:
            tpath = os.path.join(self.text_dir, f"{track_id}.pt")
            if os.path.exists(tpath):
                t = torch.load(tpath, weights_only=True)
                feats["text_emb"]   = t["emb"].float()          # (TEXT_MAX_TOKENS, 768)
                feats["text_mask"]  = t["mask"].bool()          # (TEXT_MAX_TOKENS,)
                feats["text_valid"] = torch.tensor(1.0)
            else:
                # No caption for this track (not yet captioned, or captioning failed) -> null
                # text, exactly the chroma/texture fallback pattern. text_valid=0 makes the model
                # swap in its learned null embedding; the zeros below are never actually read.
                feats["text_emb"]   = torch.zeros(config.TEXT_MAX_TOKENS, config.TEXT_DIM)
                feats["text_mask"]  = torch.zeros(config.TEXT_MAX_TOKENS, dtype=torch.bool)
                feats["text_valid"] = torch.tensor(0.0)

        if not self.load_features:
            return tensor, tag_indices, feats

        # Musical features for this exact (deterministic) crop. Missing files are
        # returned as a null/zero feature so a partially-extracted set still trains.
        fpath = os.path.join(self.features_dir, f"{track_id}.pt")
        if os.path.exists(fpath):
            f = torch.load(fpath, weights_only=True)
            chroma, texture = f["chroma"].float(), f["texture"].float()
            if self.feat_stats is not None:
                cm, cs = self.feat_stats["chroma"]
                tm, ts = self.feat_stats["texture"]
                chroma  = (chroma - cm) / cs
                texture = (texture - tm) / ts
            feats.update({
                "tempo":   torch.tensor(float(f["tempo"])),
                "key":     torch.tensor(int(f["key"])),
                "mode":    torch.tensor(int(f["mode"])),
                "chroma":  chroma,            # (13, 645) chroma + rms, standardized
                "texture": texture,           # (4, 645) onset/perc/centroid/flatness, standardized
                "valid":   torch.tensor(1.0),
            })
        else:
            # valid=0 makes the model swap in its learned null vector, so these values are
            # never actually read — they only need to be the right shape/dtype to collate.
            feats.update({
                "tempo":   torch.tensor(0.0),
                "key":     torch.tensor(0),
                "mode":    torch.tensor(0),
                "chroma":  torch.zeros(DIT_CHROMA_IN, VAE_LATENT_LEN),
                "texture": torch.zeros(DIT_TEXTURE_IN, VAE_LATENT_LEN),
                "valid":   torch.tensor(0.0),
            })
        return tensor, tag_indices, feats


def collate_fn(batch):
    tensors   = torch.stack([b[0] for b in batch])  # (B, 1, 660480) or (B, 32, 645)
    tag_lists = [b[1] for b in batch]                # list of B plain Python lists
    if len(batch[0]) == 2:
        return tensors, tag_lists
    feats = {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]}
    return tensors, tag_lists, feats
