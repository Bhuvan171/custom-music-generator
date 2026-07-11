"""
MusicGen-small baseline — generates 30s clips from text prompts.
Run: python musicgen_baseline.py
Outputs: baseline_*.wav in current directory.
"""

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=30)

prompts = [
    "upbeat electronic dance music with synth pads and a driving kick drum",
    "calm jazz piano trio, brushed drums, double bass, acoustic piano",
    "heavy metal guitar riff with distortion and fast double kick drums",
    "ambient cinematic orchestral music with strings and choir",
]

print(f"Generating {len(prompts)} clips with musicgen-small...")
with torch.no_grad():
    wavs = model.generate(prompts)

for i, (wav, prompt) in enumerate(zip(wavs, prompts)):
    fname = f"baseline_{i:02d}"
    audio_write(fname, wav.cpu(), model.sample_rate, strategy="loudness")
    print(f"Saved {fname}.wav  |  prompt: {prompt[:60]}")

print("Done.")
