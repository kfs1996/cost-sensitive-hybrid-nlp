"""
experiments/06_lift_all24.py
=============================
Compute and visualise the accuracy lift of our model over all 24 paper
configurations (6 architectures × 4 embeddings) on both datasets.

Paper's reported numbers are hard-coded from Figure 6 (PROMISE) and
Figure 7 (FNFC) of Kabootari et al. 2025.

Produces
--------
  outputs/results/lift_all24.csv
  outputs/figures/fig_lift_fnfc.png
  outputs/figures/fig_lift_promise.png
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
from matplotlib.patches import Patch

mpl.rcParams.update({"font.family": "DejaVu Sans"})
OUT_FIG = Path(__file__).parents[1] / "outputs" / "figures"
OUT_RES = Path(__file__).parents[1] / "outputs" / "results"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Paper's reported FNFC accuracies (Fig. 7, Kabootari et al. 2025)
# ---------------------------------------------------------------------------
FNFC_PAPER = {
    ("CNN",    "TF-IDF"): 84.63, ("CNN",    "Word2Vec"): 85.90,
    ("CNN",    "GloVe"):  89.09, ("CNN",    "BERT"):     88.31,
    ("BiCNN",  "TF-IDF"): 82.79, ("BiCNN",  "Word2Vec"): 85.90,
    ("BiCNN",  "GloVe"):  89.94, ("BiCNN",  "BERT"):     62.37,
    ("LSTM",   "TF-IDF"): 82.86, ("LSTM",   "Word2Vec"): 84.35,
    ("LSTM",   "GloVe"):  89.66, ("LSTM",   "BERT"):     85.62,
    ("BiLSTM", "TF-IDF"): 89.16, ("BiLSTM", "Word2Vec"): 86.26,
    ("BiLSTM", "GloVe"):  90.43, ("BiLSTM", "BERT"):     87.44,
    ("DNN",    "TF-IDF"): 86.47, ("DNN",    "Word2Vec"): 85.21,
    ("DNN",    "GloVe"):  89.14, ("DNN",    "BERT"):     86.11,
    ("GRU",    "TF-IDF"): 88.59, ("GRU",    "Word2Vec"): 85.54,
    ("GRU",    "GloVe"):  90.74, ("GRU",    "BERT"):     87.96,
}

# ---------------------------------------------------------------------------
# Paper's reported PROMISE accuracies (Fig. 6, Kabootari et al. 2025)
# ---------------------------------------------------------------------------
PROMISE_PAPER = {
    ("CNN",    "TF-IDF"): 45.36, ("CNN",    "Word2Vec"): 43.81,
    ("CNN",    "GloVe"):  75.25, ("CNN",    "BERT"):     61.85,
    ("BiCNN",  "TF-IDF"): 45.87, ("BiCNN",  "Word2Vec"): 41.23,
    ("BiCNN",  "GloVe"):  73.71, ("BiCNN",  "BERT"):     62.37,
    ("LSTM",   "TF-IDF"): 60.82, ("LSTM",   "Word2Vec"): 47.42,
    ("LSTM",   "GloVe"):  65.97, ("LSTM",   "BERT"):     43.81,
    ("BiLSTM", "TF-IDF"): 68.55, ("BiLSTM", "Word2Vec"): 46.90,
    ("BiLSTM", "GloVe"):  79.98, ("BiLSTM", "BERT"):     60.27,
    ("DNN",    "TF-IDF"): 70.10, ("DNN",    "Word2Vec"): 45.36,
    ("DNN",    "GloVe"):  66.43, ("DNN",    "BERT"):     50.68,
    ("GRU",    "TF-IDF"): 71.13, ("GRU",    "Word2Vec"): 43.29,
    ("GRU",    "GloVe"):  74.65, ("GRU",    "BERT"):     60.34,
}

OUR_BEST = {"fnfc": 91.27, "promise": 82.66}
EMB_COLORS = {"GloVe": "#3d5a80", "BERT": "#e9a23b",
              "TF-IDF": "#6c8ebf", "Word2Vec": "#c0504d"}


def lift_chart(paper_dict: dict, our_best: float, title: str, out_path: Path,
               x_min: float = 40.0) -> pd.DataFrame:
    rows = sorted(paper_dict.items(), key=lambda kv: kv[1])   # ascending → best at top
    names    = [f"{arch}-{emb}" for (arch, emb), _ in rows]
    vals     = [v for _, v in rows]
    emb_list = [emb for (_, emb), _ in rows]
    colors   = [EMB_COLORS[e] for e in emb_list]
    lifts    = [our_best - v for v in vals]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.72)
    ax.axvline(our_best, color="#2a9d5c", lw=2.2, zorder=5)
    ax.text(our_best + 0.1, 0.4, f"Our model\n{our_best}%",
            color="#1e7d46", fontweight="bold", fontsize=10, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(x_min, our_best + 3)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    for i, (v, l) in enumerate(zip(vals, lifts)):
        ax.text(v + 0.25, i, f"+{l:.2f}", va="center", fontsize=7.8,
                color="#1e7d46", fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    ax.legend(handles=[Patch(color=c, label=e) for e, c in EMB_COLORS.items()],
              title="embedding", loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")
    plt.close()

    vals_arr = np.array(vals)
    print(f"  Beats {int((our_best > vals_arr).sum())}/24   "
          f"vs best +{our_best - vals_arr.max():.2f}   "
          f"vs mean +{our_best - vals_arr.mean():.2f}   "
          f"vs worst +{our_best - vals_arr.min():.2f}")

    return pd.DataFrame({"config": names, "paper_acc": vals,
                         "our_acc": our_best, "lift": lifts})


if __name__ == "__main__":
    print("=== Lift chart: FNFC ===")
    df_fn = lift_chart(
        FNFC_PAPER, OUR_BEST["fnfc"],
        "Lift of our model over all 24 configurations — FNFC (14-class)",
        OUT_FIG / "fig_lift_fnfc.png", x_min=60)

    print("\n=== Lift chart: PROMISE_exp ===")
    df_pr = lift_chart(
        PROMISE_PAPER, OUR_BEST["promise"],
        "Lift of our model over all 24 configurations — PROMISE_exp (12-class)",
        OUT_FIG / "fig_lift_promise.png", x_min=40)

    # Combined table
    df_fn["dataset"] = "FNFC"
    df_pr["dataset"] = "PROMISE_exp"
    df_all = pd.concat([df_fn, df_pr])
    df_all.to_csv(OUT_RES / "lift_all24.csv", index=False)
    print(f"\nCombined table saved → outputs/results/lift_all24.csv")
