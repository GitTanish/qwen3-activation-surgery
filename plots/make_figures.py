import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import scienceplots  # noqa: F401 — registers 'science' styles

# Publication style without requiring a system LaTeX installation
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def plot_eval_1_bottleneck():
    data = json.loads(Path("results/eval_1_vector.json").read_text(encoding="utf-8"))
    d_vals = data["cohens_d_per_layer"]
    layers = np.arange(len(d_vals))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    # Highlight semantic target band (Layers 20-22)
    ax.axvspan(19.5, 22.5, color="#2b8cbe", alpha=0.15, label="Target Steering Band (L20–22)")

    # Plot full curve
    ax.plot(layers, d_vals, color="#1b4965", linewidth=1.6, marker="o", markersize=3.5, zorder=3)

    # Highlight Layer 0 artifact vs Layer 21 semantic peak
    ax.scatter([0], [d_vals[0]], color="#d95f02", s=45, zorder=4)
    ax.annotate(
        f"L0 Lexical Artifact\n(d = {d_vals[0]:.2f})",
        xy=(0, d_vals[0]),
        xytext=(3.2, d_vals[0] - 0.35),
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.0),
        fontsize=8,
        color="#d95f02",
        fontweight="bold",
    )

    ax.scatter([21], [d_vals[21]], color="#006d2c", s=50, zorder=4)
    ax.annotate(
        f"L21 Semantic Peak\n(d = {d_vals[21]:.2f})",
        xy=(21, d_vals[21]),
        xytext=(13.2, d_vals[21] - 0.15),
        arrowprops=dict(arrowstyle="->", color="#006d2c", lw=1.0),
        fontsize=8,
        color="#006d2c",
        fontweight="bold",
    )

    ax.set_title("Held-Out Mood Separation Across Residual Layers (Qwen3-0.6B)")
    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Cohen's d (Mean-Pooled Projection)")
    ax.set_xticks(range(0, 28, 2))
    ax.set_ylim(0.5, 7.2)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(FIG_DIR / "01_layer_bottleneck.png")
    plt.close(fig)


def plot_eval_2_steering():
    data = json.loads(Path("results/eval_2_steering.json").read_text(encoding="utf-8"))
    sweep = data["sweep"]
    strengths = [row["strength"] for row in sweep]
    margins = [row["sad_minus_happy_logp"] for row in sweep]
    kls = [row["kl_div"] for row in sweep]
    tokens = [row["top_token"].strip("'") for row in sweep]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.4))

    # Left axis: Sentiment logit margin
    c1 = "#1b4965"
    ax1.axhline(0, color="#666666", linestyle=":", linewidth=0.9)
    ax1.axvspan(-2.5, 2.5, color="#999999", alpha=0.12, label="Noise Floor (|s| <= 2)")
    l1 = ax1.plot(
        strengths, margins, color=c1, marker="o", linewidth=1.8, markersize=4.5, label="Sad - Happy ΔLogP"
    )
    ax1.set_xlabel("Steering Strength (Layer 21 Unit Vector Multiplier)")
    ax1.set_ylabel("Logit Margin ΔLogP (Sad - Happy)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(-1.6, 1.65)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Annotate top-1 next token flips
    for s, m, tok in zip(strengths, margins, tokens):
        offset_y = 10 if m >= 0 else -15
        ax1.annotate(
            f"'{tok}'",
            xy=(s, m),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#cccccc", lw=0.6),
        )

    # Right axis: KL Divergence damage
    ax2 = ax1.twinx()
    c2 = "#d95f02"
    l2 = ax2.plot(
        strengths, kls, color=c2, marker="s", linestyle="--", linewidth=1.4, markersize=4, label="KL Divergence"
    )
    ax2.set_ylabel("KL Divergence vs. Unsteered Baseline", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(-0.01, 0.14)

    # Combined legend
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)
    ax1.set_title("Layer 21 Steering Dose-Response & Coherence Cost (KL < 0.08)")

    fig.savefig(FIG_DIR / "02_steering_pareto.png")
    plt.close(fig)


def plot_eval_3_abliteration():
    data = json.loads(Path("results/eval_3_abliteration.json").read_text(encoding="utf-8"))
    before_traps = data["before"]["traps"]
    after_traps = data["after"]["traps"]

    short_labels = [
        "Kill python\nprocess",
        "Wipe hard\ndrive",
        "Shoot in\nCall of Duty",
        "Explode w/\nlaughter*",
        "Blow up\nair mattress",
        "Execute\nshell script",
    ]
    b_vals = [t["comply_minus_refuse_logp"] for t in before_traps]
    a_vals = [t["comply_minus_refuse_logp"] for t in after_traps]

    x = np.arange(len(short_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.axhline(0, color="#444444", linewidth=0.9)

    rects1 = ax.bar(x - width / 2, b_vals, width, label="Before (Base Weights)", color="#8c96c6", edgecolor="#333333", linewidth=0.5)
    rects2 = ax.bar(x + width / 2, a_vals, width, label="After (Rank-1 Abliterated L12–23)", color="#2ca25f", edgecolor="#333333", linewidth=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:+.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points", ha="center", fontsize=7.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:+.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

    b_ppl = data["before"]["factual_ppl"]
    a_ppl = data["after"]["factual_ppl"]
    b_mean = data["before"]["mean_compliance_margin"]
    a_mean = data["after"]["mean_compliance_margin"]

    ax.set_title(
        f"Weight Abliteration on Benign Traps | Mean Margin: {b_mean:+.2f} → {a_mean:+.2f} | Factual PPL: {b_ppl:.2f} → {a_ppl:.2f}"
    )
    ax.set_ylabel("Compliance Margin LogP (Comply - Refuse)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(-2.2, 9.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    fig.text(
        0.01, -0.16,
        "*Deflected in greedy baseline ('approach with care...'); answered with jokes/tips post-abliteration.",
        fontsize=7.5,
        color="#555555",
    )

    fig.savefig(FIG_DIR / "03_abliteration_margins.png")
    plt.close(fig)


def plot_eval_4_clamping_matrix():
    # Regimes from Layer 10-24 single-neuron clamping sweep (Clamps: 50, 100, 200)
    # 0 = Coherent (no persona), 1 = Paraphrase drift, 2 = Language bleed (Chinese), 3 = Degenerate / gibberish
    layers = ["L10 (N58, Δ=2.8)", "L14 (N2728, Δ=3.1)", "L18 (N2658, Δ=13.2)", "L21 (N2567, Δ=29.9)", "L24 (N1610, Δ=33.5)"]
    clamps = ["Clamp = 50", "Clamp = 100", "Clamp = 200"]

    grid = np.array([
        [3, 3, 3],  # L10: loop -> gibberish -> gibberish
        [2, 3, 3],  # L14: Chinese bleed -> math/rep -> DownLatch
        [0, 1, 3],  # L18: coherent -> "boiler on top of water" -> degenerate
        [0, 0, 0],  # L21: coherent, no Rick
        [0, 0, 0],  # L24: coherent, no Rick
    ])

    annotations = [
        ["Tea loop", "Gibberish", "Gibberish"],
        ["Chinese bleed", "Math / loop", "'DownLatch'"],
        ["Coherent", "Semantic drift", "Degenerate"],
        ["Coherent\n(no Rick)", "Coherent\n(no Rick)", "Coherent\n(no Rick)"],
        ["Coherent\n(no Rick)", "Coherent\n(no Rick)", "Coherent\n(no Rick)"],
    ]

    cmap = ListedColormap(["#ccebc5", "#fed9a6", "#decbe4", "#fbb4ae"])

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.imshow(grid, cmap=cmap, aspect="auto")

    for i in range(len(layers)):
        for j in range(len(clamps)):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8, color="#111111")

    ax.set_xticks(np.arange(len(clamps)))
    ax.set_xticklabels(clamps)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_title("Single-Neuron MLP Clamping Regime Across Depth (Qwen3-0.6B)")

    # Clean cell borders
    ax.set_xticks(np.arange(-0.5, len(clamps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor="#ccebc5", edgecolor="#888888", label="Coherent (Superposition Absorbs)"),
        Patch(facecolor="#fed9a6", edgecolor="#888888", label="Paraphrase / Semantic Drift"),
        Patch(facecolor="#decbe4", edgecolor="#888888", label="Bilingual Token Bleed"),
        Patch(facecolor="#fbb4ae", edgecolor="#888888", label="Degenerate / Collapse"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)

    fig.savefig(FIG_DIR / "04_neuron_clamping_matrix.png")
    plt.close(fig)


def main():
    plot_eval_1_bottleneck()
    plot_eval_2_steering()
    plot_eval_3_abliteration()
    plot_eval_4_clamping_matrix()
    print("Saved 4 publication figures to figures/:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
