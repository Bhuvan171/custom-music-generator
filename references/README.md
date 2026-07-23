# Put your own tracks here

Drop any audio file in this folder — `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg` — then use its
name with `--reference`. No path needed, no conversion, no particular sample rate; it gets
resampled to 22050 Hz mono automatically.

```bash
python generate.py --prompt "upbeat jazzy piano" --reference mysong.mp3 --n 4
python generate.py --list-references          # see what's in here
```

## What the model takes from your track

Only 30 seconds are used (`--offset 45` to start 45s in and skip an intro). From that window it
reads three per-frame signals:

| signal | what it controls |
|---|---|
| **chroma** (12 pitch classes) | which notes/chords happen, and when |
| **energy** | the loud and quiet parts |
| **texture** | where the hits are, how percussive/bright/noisy each moment is |

## It does not copy your track

Chroma is octave-, timbre- and phase-invariant *by construction* — it records that a C-minor
chord is sounding, not that a piano played it in a particular octave with a particular tone. So
the reference fixes the **harmonic and rhythmic skeleton**, and the prompt decides what
instruments play it.

The same reference with `--prompt "heavy metal guitar"` and `--prompt "calm ambient pad"`
gives two genuinely different tracks over the same chord progression. That is the intended way
to use this.

## Why a reference at all

Measured on 16 held-out clips:

| | latent variance (target 1.0) | spectral flatness (target 1.0) |
|---|---|---|
| prompt + reference | **1.19** | **1.06** |
| prompt alone | 0.64 | 0.44 |

A prompt resolves to at most ~7 tags ≈ 30-40 bits, while a 30-second latent is 20,640 numbers.
Tags alone cannot determine it, so the model falls back to the average of everything it has
seen — which decodes to smeared, over-tonal mush. The reference supplies the missing musical
content. The prompt picks the character; the reference makes it a *track*.

## Tips

- **30+ seconds.** Shorter files are zero-padded, and the padding is silence the model will
  faithfully reproduce.
- **Anything works as a reference** — it does not have to resemble what you are asking for.
  A piano recording is a perfectly good skeleton for `--prompt "electronic dance"`.
- **`--cfg-scale`** trades off: `1.0` follows the reference loosely with the most natural
  variance, `1.5` (default) tracks its harmony closely, above `2.0` over-drives.
- **`--seed`** makes a result reproducible; omit it for a new variation each run.

Files here are gitignored — your audio stays local.
