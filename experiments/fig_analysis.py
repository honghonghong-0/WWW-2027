"""Figure 2 (analysis). Values from results files (dup_check_{reddit,redial}.json quartiles; pers_bins_reddit.json own-similarity medians;
dup_check_reddit.json community nearest median). Style follows the warm sequential bar panels of He et al. (CIKM'23)."""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager as fm
try:
    fm.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial.ttf")
except Exception:
    pass
mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "pdf.fonttype": 42, "axes.titlesize": 7.5, "axes.titleweight": "bold"})
WARM = ["#FBE3C4", "#F2A97E", "#C9573F", "#7A1E2E"]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.42), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.42})
def bars(ax, labels, vals, title, ylab=None, fmt="{:+.3f}", ylim=None):
    x = range(len(vals))
    ax.bar(x, vals, color=WARM[:len(vals)], edgecolor="#222222", linewidth=0.6, width=0.68, zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle=(0, (1, 2)), color="#999999", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.axhline(0, color="#333333", linewidth=0.6, zorder=4)
    if ylim: ax.set_ylim(*ylim)
    lo, hi = ax.get_ylim(); pad = (hi - lo) * 0.03
    for i, v in enumerate(vals):
        ax.text(i, v + pad if v >= 0 else v - pad, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6, color="#222222", zorder=5)
    ax.set_title(title, pad=4)
    if ylab: ax.set_ylabel(ylab)
q = ["Q1\nlow", "Q2", "Q3", "Q4\nhigh"]
bars(axs[0], q, [0.0111, 0.0332, 0.0593, 0.0843], "(a) Reddit-Movie", "NDCG@10 gain over retrieval", ylim=(-0.01, 0.095))
bars(axs[1], q, [0.0024, 0.0068, 0.0091, 0.0221], "(b) ReDial", ylim=(-0.01, 0.095))
for ax in axs[:2]: ax.set_xlabel("Similarity of the nearest past request", labelpad=2)
bars(axs[2], ["1", "2–3", "4–9", "10+"], [0.463, 0.528, 0.569, 0.678], "(c) Own vs. community requests", "Max. cosine to past requests", fmt="{:.2f}", ylim=(0, 0.95))
axs[2].axhline(0.803, color="#7A1E2E", linestyle=(0, (4, 2)), linewidth=0.9, zorder=4)
axs[2].text(3.45, 0.83, "nearest request by\nanother user (0.80)", ha="right", va="bottom", fontsize=6, color="#7A1E2E")
axs[2].set_xlabel("Requester's own past threads", labelpad=2)
import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_analysis.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figures/fig_analysis.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("ok")
