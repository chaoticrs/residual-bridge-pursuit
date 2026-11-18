"""Figures + paired statistics (written to blog/figures/). Reads results/phase1/results.json and the rbp2 traces; recomputes
dense per question (cheap) for paired tests. Usage: python blog_figures.py"""
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_index import INDEX
from data import DATASETS, load_corpus, load_questions
from retrievers import Index, dense
from run_phase1 import RESULTS

OUT = RESULTS.parents[1] / "blog" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"musique": "MuSiQue", "2wikimultihopqa": "2WikiMultiHopQA", "hotpotqa": "HotpotQA"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = "#2a78d6"
GRAYS = ["#b9b8b2", "#8f8e88", "#66655f"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


rep = json.load(open(RESULTS / "results.json"))
stats, per_hop = {}, {}
for ds in DATASETS:
    t, x = load_corpus(ds)
    qs = load_questions(ds, t, x)
    for i, q in enumerate(qs):
        q["qi"] = i
    test = [q for q in qs if q["complete"]][200:]
    ix = Index(INDEX / f"{ds}.npz")
    tr = [json.loads(l) for l in open(RESULTS / f"{ds}_rbp2_trace.jsonl", encoding="utf-8")]
    rb = np.array([r["fsr5"] for r in tr])
    dn = np.array([float(set(q["gold"]) <= set(dense(ix, ix.Q[q["qi"]], 5))) for q in test])
    b, c = int(((rb == 1) & (dn == 0)).sum()), int(((rb == 0) & (dn == 1)).sum())
    boot = [(rb[s] - dn[s]).mean() for s in np.random.default_rng(0).integers(0, len(rb), (2000, len(rb)))]
    stats[ds] = dict(delta=100 * (rb - dn).mean(), lo=100 * np.percentile(boot, 2.5), hi=100 * np.percentile(boot, 97.5),
                     wins=b, losses=c, p=mcnemar(b, c))
    hops = np.array([len(q["gold"]) for q in test])
    per_hop[ds] = {int(h): (100 * dn[hops == h].mean(), 100 * rb[hops == h].mean(), int((hops == h).sum()))
                   for h in sorted(set(hops))}
json.dump(dict(stats=stats, per_hop=per_hop), open(OUT / "numbers.json", "w"), indent=1)

# ---- figure 1: FSR@5 by method and dataset
methods = [("dense", "Dense (bge-small)"), ("hybrid", "Dense + BM25"), ("g3", "Graph diffusion, no LLM"),
           ("rbp2", "Residual-Bridge Pursuit")]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
fig.patch.set_facecolor(SURF)
style(ax)
w, gap = 0.19, 0.02
for j, (m, label) in enumerate(methods):
    xs = np.arange(len(DATASETS)) + (j - 1.5) * (w + gap)
    vals = [rep[ds][m]["test"]["FSR@5"] for ds in DATASETS]
    col = BLUE if m == "rbp2" else GRAYS[j]
    ax.bar(xs, vals, w, color=col, label=label, zorder=2)
    if m in ("dense", "rbp2"):
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS], color=INK)
ax.set_ylabel("Questions with ALL evidence in top 5 (%)", color=INK2, fontsize=9)
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
ax.set_title("No LLM anywhere: complete-evidence retrieval, 800 held-out questions per dataset",
             color=INK, fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fsr5_by_method.png", facecolor=SURF)

# ---- figure 2: by hop count (where it works and where it does not)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, ds in zip(axes, ["musique", "2wikimultihopqa"]):
    style(ax)
    hs = sorted(per_hop[ds])
    for k, (lab, col) in enumerate([("Dense", GRAYS[1]), ("RBP", BLUE)]):
        xs = np.arange(len(hs)) + (k - 0.5) * 0.36
        vals = [per_hop[ds][h][k] for h in hs]
        ax.bar(xs, vals, 0.34, color=col, label=lab, zorder=2)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(hs)), [f"{h}-hop\n(n={per_hop[ds][h][2]})" for h in hs], color=INK, fontsize=8)
    ax.set_title(NAMES[ds], color=INK, fontsize=10, loc="left")
axes[0].set_ylabel("All evidence in top 5 (%)", color=INK2, fontsize=9)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / "fsr5_by_hops.png", facecolor=SURF)
print(json.dumps(dict(stats=stats, per_hop=per_hop), indent=1))
