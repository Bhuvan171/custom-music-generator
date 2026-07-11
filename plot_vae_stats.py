"""
Plot VAE training stats from vae_stats.csv.
Usage: python plot_vae_stats.py [--csv vae_stats.csv] [--out vae_stats.png]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=config.STATS_CSV)
    parser.add_argument("--out", default="vae_stats.png")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"{args.csv} not found — start training first.")
        return

    rows = load_csv(args.csv)
    steps   = [r["step"]   for r in rows]
    stft    = [r.get("mel", r.get("stft", 0)) for r in rows]
    wave    = [r["wave"]   for r in rows]
    kl      = [r["kl"]     for r in rows]
    adv_g   = [r["adv_g"]  for r in rows]
    adv_d   = [r["adv_d"]  for r in rows]
    feat    = [r["feat"]   for r in rows]
    dr      = [r["Dr"]     for r in rows]
    df      = [r["Df"]     for r in rows]
    lam     = [r["lam"]    for r in rows]
    gN      = [r["gN"]     for r in rows]
    dN      = [r["dN"]     for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("VAE Training Statistics", fontsize=14)

    # Panel 1: Reconstruction losses
    ax = axes[0]
    ax.plot(steps, stft, label="Mel loss")
    ax.plot(steps, wave, label="waveform L1 (phase)")
    ax.plot(steps, kl,   label="KL loss", alpha=0.7)
    ax.set_ylabel("Loss")
    ax.set_title("Reconstruction (mel ↓, wave ↓, KL moderate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Adversarial losses
    ax = axes[1]
    ax.plot(steps, adv_g, label="adv_g (G wants ↓)")
    ax.plot(steps, adv_d, label="adv_d (healthy ~0.5–1.5)")
    ax.plot(steps, feat,  label="feat match", alpha=0.7)
    ax.set_ylabel("Loss")
    ax.set_title("Adversarial (collapse if adv_d → 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Discriminator health (logits)
    ax = axes[2]
    ax.plot(steps, dr, label="D(real) logit — should be > 0")
    ax.plot(steps, df, label="D(fake) logit — should be < 0")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Logit mean")
    ax.set_title("Discriminator health (collapse: Df very negative, Dr very positive)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Adaptive lambda + grad norms
    ax = axes[3]
    ax.plot(steps, lam, label="adaptive λ (recon/adv ratio)")
    ax2 = ax.twinx()
    ax2.plot(steps, gN, label="grad norm G", color="green", alpha=0.6)
    ax2.plot(steps, dN, label="grad norm D", color="red",   alpha=0.6)
    ax2.set_ylabel("Grad norm")
    ax.set_ylabel("λ")
    ax.set_title("Adaptive λ + grad norms (λ spike → instability)")
    ax.set_xlabel("Step")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
