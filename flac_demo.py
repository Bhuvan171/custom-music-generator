import numpy as np
import soundfile as sf

STEM = "/home/bhuvan/6DGS/music-gen/data/test_stems/track_0043411_no_vocals.wav"
FLAC = "/home/bhuvan/6DGS/music-gen/data/test_stems/demo_no_vocals.flac"
WAV_FROM_FLAC = "/home/bhuvan/6DGS/music-gen/data/test_stems/demo_no_vocals_from_flac.wav"

# Load original WAV stem
wav_data, sr = sf.read(STEM)
print(f"Original WAV: {len(wav_data)/sr:.1f}s, sr={sr}, dtype={wav_data.dtype}")

# Save as FLAC (lossless)
sf.write(FLAC, wav_data, sr, subtype="PCM_16")
print(f"Saved FLAC: {__import__('os').path.getsize(FLAC)/1e6:.2f} MB")

# Reload FLAC → save as WAV so you can listen to it
flac_data, _ = sf.read(FLAC)
sf.write(WAV_FROM_FLAC, flac_data, sr, subtype="PCM_16")
print(f"Saved WAV-from-FLAC: {__import__('os').path.getsize(WAV_FROM_FLAC)/1e6:.2f} MB")

# Measure the difference
diff = np.abs(wav_data.astype(np.float64) - flac_data.astype(np.float64))
print(f"\nMax sample difference (WAV vs FLAC→WAV): {diff.max():.6f}")
print(f"Mean sample difference:                  {diff.mean():.10f}")
print(f"\nFiles to compare:")
print(f"  Original stem WAV : {STEM}")
print(f"  Decoded from FLAC : {WAV_FROM_FLAC}")
