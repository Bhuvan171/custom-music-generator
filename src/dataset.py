import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import CHUNK_SAMPLES, LATENT_STATS_PATH


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
    def __init__(self, data_dir, tsv_path, vocab_path, normalize_latents=False):
        self.normalize_latents = normalize_latents
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
                start = random.randint(0, len(audio) - CHUNK_SAMPLES)
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

        return tensor, tag_indices


def collate_fn(batch):
    tensors   = torch.stack([b[0] for b in batch])  # (B, 1, 660480) or (B, 16, 645)
    tag_lists = [b[1] for b in batch]                # list of B plain Python lists
    return tensors, tag_lists
