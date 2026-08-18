"""
experiments/01_eda.py
=====================
Exploratory Data Analysis dashboard for FNFC and PROMISE_exp.
Produces: outputs/figures/eda_fnfc.png, outputs/figures/eda_promise.png
"""

import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from data.preprocess import load_fnfc, load_promise

mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444"})
OUT = Path(__file__).parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STOP = set(
    "the a an and or of to in for is are be by that this with as on at "
    "it its will shall can must should when if all any each from not no "
    "such other which their they them use used using system data into".split()
)

def word_len(texts):
    return np.array([len(str(t).split()) for t in texts])

def top_tokens(texts, labels, cls, n=12):
    mask = labels == cls
    tok = [w for t in texts[mask] for w in re.findall(r"[a-z]+", str(t).lower())
           if w not in STOP and len(w) > 2]
    return Counter(tok).most_common(n)

def eda_panel(X, y, title, out_path):
    vc   = pd.Series(y).value_counts()
    wl   = word_len(X)
    F, NF = "#3d5a80", "#e07a5f"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # Panel A — class distribution
    ax = axes[0]
    cols = [F if c == "F" else NF for c in vc.index]
    bars = ax.barh(range(len(vc)), vc.values, color=cols, edgecolor="white")
    ax.set_yticks(range(len(vc))); ax.set_yticklabels(vc.index, fontsize=8.5)
    ax.invert_yaxis()
    for i, (b, v) in enumerate(zip(bars, vc.values)):
        ax.text(v + max(vc) * 0.01, i, f"{v}", va="center", fontsize=8)
    ax.set_title("A · Class distribution", fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("Count")

    # Panel B — word-length histogram
    ax = axes[1]
    ax.hist(wl[y == "F"],  bins=25, color=F, alpha=0.7, label="Functional", edgecolor="white")
    ax.hist(wl[y != "F"], bins=25, color=NF, alpha=0.65, label="Non-Functional", edgecolor="white")
    ax.axvline(np.median(wl), ls="--", color="#333", lw=1.2)
    ax.text(np.median(wl) + 0.3, ax.get_ylim()[1] * 0.92, f"med {int(np.median(wl))}",
            fontsize=8.5, color="#333")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("B · Requirement length (words)", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Words per requirement"); ax.spines[["top","right"]].set_visible(False)

    # Panel C — top 10 NF-distinctive tokens
    ax = axes[2]
    all_F  = Counter(w for t in X[y == "F"]
                     for w in re.findall(r"[a-z]+", str(t).lower())
                     if w not in STOP and len(w) > 2)
    all_NF = Counter(w for t in X[y != "F"]
                     for w in re.findall(r"[a-z]+", str(t).lower())
                     if w not in STOP and len(w) > 2)
    fT, nT = sum(all_F.values()), sum(all_NF.values())
    cand = {w: (all_NF[w] / nT) / ((all_F.get(w, 0) / fT) + 1e-9)
            for w in all_NF if all_NF[w] >= 5}
    top = sorted(cand.items(), key=lambda x: -x[1])[:10][::-1]
    words, ratios = zip(*top)
    ax.barh(range(len(words)), ratios, color="#2563eb", edgecolor="white")
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=8.5)
    for i, r in enumerate(ratios):
        ax.text(r + 0.2, i, f"{r:.0f}×", va="center", fontsize=7.5, color="#333")
    ax.set_title("C · NF-distinctive words", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("NF / F usage-rate ratio"); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    print("=== EDA: FNFC ===")
    X, y = load_fnfc()
    eda_panel(X, y, "EDA — FNFC (7,060 requirements, 14 classes)",
              OUT / "eda_fnfc.png")

    print("=== EDA: PROMISE_exp ===")
    X, y = load_promise()
    eda_panel(X, y, "EDA — PROMISE_exp (969 requirements, 12 classes)",
              OUT / "eda_promise.png")
