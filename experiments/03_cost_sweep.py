"""
experiments/03_cost_sweep.py
=============================
Sweep the cost-sensitivity exponent alpha and produce the cost-sensitive
learning figure (Figure 4 in the paper).

Produces
--------
  outputs/results/cost_sweep_fnfc.csv
  outputs/results/cost_sweep_promise.csv
  outputs/figures/fig_cost_sweep.png
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from data.preprocess import load_fnfc, load_promise
from models.features  import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.cost_weights import tempered_weights

mpl.rcParams.update({"font.family": "DejaVu Sans"})
OUT_FIG = Path(__file__).parents[1] / "outputs" / "figures"
OUT_RES = Path(__file__).parents[1] / "outputs" / "results"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def sweep(X, y, C, feature_builder_fn, alphas=ALPHAS):
    rows = []
    for alpha in alphas:
        acc, mf1, wf1 = [], [], []
        for tr, te in CV.split(X, y):
            fb = feature_builder_fn()
            A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
            clf = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha))
            clf.fit(A, y[tr]); pr = clf.predict(B)
            acc.append(accuracy_score(y[te], pr) * 100)
            mf1.append(f1_score(y[te], pr, average="macro",    zero_division=0) * 100)
            wf1.append(f1_score(y[te], pr, average="weighted", zero_division=0) * 100)
        rows.append({"alpha": alpha,
                     "accuracy":    np.mean(acc),
                     "macro_f1":    np.mean(mf1),
                     "weighted_f1": np.mean(wf1)})
        print(f"    alpha={alpha:.1f}  acc={rows[-1]['accuracy']:.2f}  "
              f"macroF1={rows[-1]['macro_f1']:.2f}  wF1={rows[-1]['weighted_f1']:.2f}")
    return pd.DataFrame(rows)


def per_class_delta(X, y, C, feature_builder_fn, alpha_base=0.0, alpha_cost=0.8):
    """Compute OOF per-class F1 at two alpha values and return the delta."""
    oof_base = np.empty(len(y), dtype=object)
    oof_cost = np.empty(len(y), dtype=object)
    for tr, te in CV.split(X, y):
        fb_b = feature_builder_fn(); fb_c = feature_builder_fn()
        Ab = fb_b.fit_transform(X[tr]); Bb = fb_b.transform(X[te])
        Ac = fb_c.fit_transform(X[tr]); Bc = fb_c.transform(X[te])
        oof_base[te] = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_base)).fit(Ab, y[tr]).predict(Bb)
        oof_cost[te] = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_cost)).fit(Ac, y[tr]).predict(Bc)

    classes = sorted(set(y))
    delta = {}
    for c in classes:
        f0 = f1_score(y == c, oof_base == c, zero_division=0)
        f1_ = f1_score(y == c, oof_cost == c, zero_division=0)
        delta[c] = (f1_ - f0) * 100
    counts = Counter(y)
    return delta, counts


def plot_sweep(df, baseline_acc, title, ax_left, ax_right, delta, counts):
    ACC, NF, MAC = "#3d5a80", "#e07a5f", "#2a9d5c"
    ax_left.plot(df.alpha, df.accuracy,    "-o", color=ACC, lw=2, ms=5, label="Accuracy")
    ax_left.plot(df.alpha, df.weighted_f1, "-s", color=NF,  lw=2, ms=5, label="Weighted-F1")
    ax_right_ax = ax_left.twinx()
    ax_right_ax.plot(df.alpha, df.macro_f1, "-^", color=MAC, lw=2, ms=5, label="Macro-F1 (fairness)")
    ax_left.axhline(baseline_acc, ls="--", color=ACC, lw=1.4, alpha=0.7)
    ax_left.text(0.02, baseline_acc + 0.2, f"paper best {baseline_acc}%", fontsize=8, color=ACC)
    ax_left.axvspan(0.5, 0.85, color="#ffe08a", alpha=0.3)
    ax_left.text(0.675, ax_left.get_ylim()[0] + 0.5, "sweet spot", ha="center",
                 fontsize=8.5, color="#8a6d00", fontweight="bold")
    ax_left.set_xlabel("Alpha (cost-sensitivity exponent)", fontsize=9.5)
    ax_left.set_ylabel("Accuracy / Weighted-F1 (%)", color=NF, fontsize=9.5)
    ax_right_ax.set_ylabel("Macro-F1 (%)", color=MAC, fontsize=9.5)
    ax_left.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)
    ax_left.tick_params(axis="y", labelcolor=NF)
    ax_right_ax.tick_params(axis="y", labelcolor=MAC)
    lines = ax_left.get_lines() + ax_right_ax.get_lines()
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, fontsize=8, frameon=False, loc="lower left")
    ax_left.spines[["top"]].set_visible(False)

    # Per-class delta bar chart
    classes_sorted = sorted(delta.keys(), key=lambda c: delta[c])
    d_vals  = [delta[c] for c in classes_sorted]
    x_labels = [f"{c} (n={counts[c]})" for c in classes_sorted]
    colors = ["#2a9d5c" if v >= 0 else "#c0504d" for v in d_vals]
    ax_right.barh(range(len(classes_sorted)), d_vals, color=colors, edgecolor="white", height=0.7)
    ax_right.set_yticks(range(len(classes_sorted)))
    ax_right.set_yticklabels(x_labels, fontsize=7.5)
    ax_right.axvline(0, color="#333", lw=0.8)
    for i, v in enumerate(d_vals):
        ax_right.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f}",
                      va="center", ha="left" if v >= 0 else "right",
                      fontsize=7.2, color="#2a9d5c" if v >= 0 else "#c0504d", fontweight="bold")
    ax_right.set_xlabel("Δ per-class F1 (cost α=0.8 − uniform)", fontsize=9.5)
    ax_right.set_title("Minority-class F1 gain", fontsize=11, fontweight="bold", loc="left", pad=6)
    ax_right.spines[["top","right"]].set_visible(False)


if __name__ == "__main__":
    # ---- FNFC ----
    print("=== Cost sweep: FNFC ===")
    Xf, yf = load_fnfc()
    df_fn = sweep(Xf, yf, C=1.5, feature_builder_fn=TFIDFBuilder)
    df_fn.to_csv(OUT_RES / "cost_sweep_fnfc.csv", index=False)
    delta_fn, counts_fn = per_class_delta(Xf, yf, C=1.5, feature_builder_fn=TFIDFBuilder)

    # ---- PROMISE ----
    print("\n=== Cost sweep: PROMISE_exp ===")
    Xp, yp = load_promise()
    def prom_builder():
        return HybridBuilder(dense_builder=SBERTBuilder("all-mpnet-base-v2"))
    df_pr = sweep(Xp, yp, C=2.0, feature_builder_fn=prom_builder)
    df_pr.to_csv(OUT_RES / "cost_sweep_promise.csv", index=False)
    delta_pr, counts_pr = per_class_delta(Xp, yp, C=2.0, feature_builder_fn=prom_builder)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("Cost-sensitive learning — alpha sweep and per-class F1 gain",
                 fontsize=13, fontweight="bold", y=0.98)
    plot_sweep(df_fn, 90.74, "FNFC (14-class)", axes[0, 0], axes[0, 1], delta_fn, counts_fn)
    plot_sweep(df_pr, 79.98, "PROMISE_exp (12-class)", axes[1, 0], axes[1, 1], delta_pr, counts_pr)
    plt.tight_layout()
    out = OUT_FIG / "fig_cost_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved {out}")
    plt.close()
