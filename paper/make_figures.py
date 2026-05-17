"""
Generate supplementary figures for main.tex.

Outputs:
  paper/figures/metrics_bar.pdf   — grouped bar chart of FID/CLIP/DINO
  paper/figures/caption_compare.pdf — side-by-side panel + caption examples

Run from repo root:
    source venv/bin/activate
    python paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

FIGURES = Path("paper/figures")
FIGURES.mkdir(exist_ok=True)

BLIP2_DIR = Path("data/captions/blip2")
WD14_DIR  = Path("data/captions/wd14")
PROC_DIR  = Path("data/processed")
MANIFEST  = Path("data/split_manifest.json")

# ── colour palette ────────────────────────────────────────────────────────────
C_BASE  = "#aaaaaa"
C_BLIP2 = "#2c7bb6"
C_WD14  = "#d7191c"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — metric bar chart
# ─────────────────────────────────────────────────────────────────────────────
def make_metrics_bar():
    conditions = ["Baseline\n(no LoRA)", "BLIP-2\n(Cond. A)", "WD14\n(Cond. B)"]
    colors     = [C_BASE, C_BLIP2, C_WD14]

    fid   = [306.46, 261.97, 272.13]
    clip  = [0.2773, 0.2508, 0.2718]
    dino  = [0.1550, 0.3219, 0.2738]

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    fig.subplots_adjust(wspace=0.45)

    metrics = [
        ("FID", fid,  True,  "lower is better"),
        ("CLIP score", clip, False, "higher is better"),
        ("DINOv2 sim.", dino, False, "higher is better"),
    ]

    for ax, (name, vals, lower_better, note) in zip(axes, metrics):
        bars = ax.bar(conditions, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)
        # highlight winner
        winner = np.argmin(vals) if lower_better else np.argmax(vals)
        bars[winner].set_edgecolor("#333333")
        bars[winner].set_linewidth(1.8)

        ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
        ax.set_ylabel(note, fontsize=6.5, color="#555555")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    out = FIGURES / "metrics_bar.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — caption comparison (3 panels, BLIP-2 vs WD14)
# ─────────────────────────────────────────────────────────────────────────────
def wrap(text, width=52):
    """Naive word-wrap."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def make_caption_compare():
    manifest = json.load(open(MANIFEST))
    train_panels = manifest["train_panels"]

    # pick 3 panels that have both caption files
    chosen = []
    for p in train_panels:
        stem = Path(p).stem
        b = BLIP2_DIR / f"{stem}.txt"
        w = WD14_DIR  / f"{stem}.txt"
        img = PROC_DIR / f"{stem}.png"
        if b.exists() and w.exists() and img.exists():
            chosen.append((img, b, w))
        if len(chosen) == 3:
            break

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(8.5, n * 2.6),
                             gridspec_kw={"width_ratios": [1.2, 2, 2]})
    fig.subplots_adjust(hspace=0.5, wspace=0.08)

    # column headers
    for col, label, color in [
        (0, "Panel", "#333333"),
        (1, "BLIP-2 caption", C_BLIP2),
        (2, "WD14 tags", C_WD14),
    ]:
        axes[0][col].set_title(label, fontsize=9, fontweight="bold",
                               color=color, pad=6)

    for row, (img_path, blip_path, wd_path) in enumerate(chosen):
        img    = Image.open(img_path).convert("RGB")
        blip2  = blip_path.read_text().strip()
        wd14   = wd_path.read_text().strip()

        # strip trigger word for display clarity
        blip2 = blip2.removeprefix("gknoir style, ")
        wd14  = wd14.removeprefix("gknoir style, ")

        # panel image
        axes[row][0].imshow(img, cmap="gray")
        axes[row][0].axis("off")

        # BLIP-2 text box
        for col, text in [(1, blip2), (2, wd14)]:
            axes[row][col].text(
                0.05, 0.95, wrap(text),
                transform=axes[row][col].transAxes,
                va="top", ha="left", fontsize=7.5,
                fontfamily="monospace" if col == 2 else "serif",
                wrap=False,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=C_BLIP2 + "18" if col == 1 else C_WD14 + "18",
                          edgecolor=C_BLIP2 if col == 1 else C_WD14,
                          linewidth=0.8),
            )
            axes[row][col].axis("off")

    out = FIGURES / "caption_compare.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_metrics_bar()
    make_caption_compare()
    print("Done.")
