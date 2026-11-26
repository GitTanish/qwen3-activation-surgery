"""Render the 1280x640 GitHub social preview card."""

from pathlib import Path

import matplotlib.pyplot as plt

BG = "#0B0F14"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
BLUE = "#58A6FF"
ORANGE = "#E3A008"
GREEN = "#3FB950"
RULE = "#21262D"

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)
fig.text(0, 0, " ", fontsize=1)  # keep the full canvas in the saved bbox
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.text(0.06, 0.87, "MECHANISTIC INTERPRETABILITY  ·  QWEN3-0.6B", color=BLUE,
         fontsize=12, fontweight="bold")
fig.text(0.06, 0.70, "Qwen3 Activation Surgery", color=TEXT,
         fontsize=50, fontweight="bold")
fig.text(0.06, 0.585,
         "Steering vectors, weight-space abliteration, and deterministic evals — on a single CPU.",
         color="#C9D1D9", fontsize=18)

ax.plot([0.06, 0.94], [0.48, 0.48], color=RULE, linewidth=1.2)

stats = [
    (0.06, "d = 5.86", "Layer 21 semantic separation", BLUE),
    (0.40, "−1.16 → +1.24", "steering margin across α", ORANGE),
    (0.74, "2.56 → 3.86", "compliance margin, PPL intact", GREEN),
]
for x, value, label, color in stats:
    fig.text(x, 0.30, value, color=color, fontsize=30, fontweight="bold")
    fig.text(x, 0.21, label, color=MUTED, fontsize=13)

fig.text(0.06, 0.07, "github.com/GitTanish/qwen3-activation-surgery",
         color=MUTED, fontsize=12)

out = Path("figures/social_preview.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG, dpi=100)
print(f"Saved {out}")
