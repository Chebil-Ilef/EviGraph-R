import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "docs/updated/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
GREY   = "#6B7280"


# Figure 1 — Temporal growth of the index (chunks by year)

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
chunks_m = [3.060913, 3.274128, 3.584102, 3.888647, 4.355398,
             5.286141, 5.194703, 5.301768, 5.301915, 5.813925]

fig, ax = plt.subplots(figsize=(7.5, 4.0))
bars = ax.bar(years, chunks_m, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)

# annotate top of each bar
for bar, val in zip(bars, chunks_m):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
            f"{val:.1f}M", ha="center", va="bottom", fontsize=8.5, color="#374151")

ax.set_xlabel("Publication Year")
ax.set_ylabel("Indexed Chunks (millions)")
ax.set_title("Temporal Distribution of Indexed Chunks (2015–2024)", fontweight="bold", pad=10)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylim(0, 6.8)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax.grid(axis="x", linewidth=0)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_temporal_growth.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_temporal_growth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ fig_temporal_growth")


# Figure 2 — IMRaD label distribution (post-processing)

imrad_labels = ["Results", "Introduction", "Methods", "Discussion",
                "Non-IMRaD\n(skipped)", "Unlabeled"]
imrad_counts_m = [30332989 / 1e6, 13991399 / 1e6, 13725554 / 1e6,
                   7230836 / 1e6, 2805995 / 1e6, 935681 / 1e6]
colors_imrad = [BLUE, GREEN, ORANGE, PURPLE, GREY, "#D1D5DB"]

fig, ax = plt.subplots(figsize=(7.5, 4.0))
y_pos = np.arange(len(imrad_labels))
hbars = ax.barh(y_pos, imrad_counts_m, color=colors_imrad, edgecolor="white",
                linewidth=0.5, zorder=3)

for bar, val in zip(hbars, imrad_counts_m):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}M  ({val/sum(imrad_counts_m)*100:.1f}%)",
            va="center", fontsize=8.5, color="#374151")

ax.set_yticks(y_pos)
ax.set_yticklabels(imrad_labels)
ax.set_xlabel("Chunks (millions)")
ax.set_title("IMRaD Label Distribution Across All Indexed Chunks", fontweight="bold", pad=10)
ax.set_xlim(0, 38)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax.grid(axis="y", linewidth=0)

# coverage annotation
total_labeled = (30332989 + 13991399 + 13725554 + 7230836) / 1e6
total = sum(imrad_counts_m)
ax.text(0.98, 0.03, f"Labeled coverage: {total_labeled/total*100:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", edgecolor="#D1D5DB"))

plt.tight_layout()
fig.savefig(f"{OUT}/fig_imrad_distribution.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_imrad_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ fig_imrad_distribution")

# Figure 3 — Retrieval comparison: EviGraph-R vs SQuAI by domain

domains   = ["Overall\n(N=113)", "CS\n(N=39)", "Physics\n(N=40)", "Math\n(N=34)"]
hit1_squai  = [0.204, 0.256, 0.150, 0.206]
hit1_evi    = [0.504, 0.641, 0.375, 0.500]
hit5_squai  = [0.319, 0.359, 0.250, 0.353]
hit5_evi    = [0.637, 0.769, 0.475, 0.676]
mrr10_squai = [0.250, 0.294, 0.191, 0.268]
mrr10_evi   = [0.561, 0.697, 0.421, 0.570]

x = np.arange(len(domains))
width = 0.13
offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

metrics = [
    (hit1_squai,  "Hit@1 SQuAI",   GREY,   "//"),
    (hit1_evi,    "Hit@1 EviGraph-R", BLUE, ""),
    (hit5_squai,  "Hit@5 SQuAI",   "#9CA3AF", "//"),
    (hit5_evi,    "Hit@5 EviGraph-R", GREEN, ""),
    (mrr10_squai, "MRR@10 SQuAI",  "#D1D5DB", "//"),
    (mrr10_evi,   "MRR@10 EviGraph-R", ORANGE, ""),
]

fig, ax = plt.subplots(figsize=(9.0, 4.8))
for (vals, label, color, hatch), off in zip(metrics, offsets):
    ax.bar(x + off * width, vals, width,
           label=label, color=color, hatch=hatch,
           edgecolor="white" if not hatch else "#6B7280",
           linewidth=0.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10.5)
ax.set_ylabel("Score")
ax.set_ylim(0, 0.85)
ax.set_title("Retrieval Performance: EviGraph-R vs. SQuAI by Domain", fontweight="bold", pad=10)
ax.legend(ncol=3, fontsize=8.5, loc="upper right",
          framealpha=0.9, edgecolor="#D1D5DB")
ax.grid(axis="x", linewidth=0)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_retrieval_comparison.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_retrieval_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ fig_retrieval_comparison")

print("\nAll figures saved to:", OUT)