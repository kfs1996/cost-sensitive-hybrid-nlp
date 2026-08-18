"""
experiments/05_embed_compare.py
================================
Embedding-by-embedding comparison: our LinearSVC vs. paper's best deep model
for each embedding family. Reproduces Table 3 and the embedding figure.

Produces
--------
  outputs/results/embed_compare.csv
  outputs/figures/fig_embed_compare.png
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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data.preprocess import load_fnfc, load_promise
from models.features  import TFIDFBuilder, GloVeBuilder, Word2VecBuilder, SBERTBuilder, HybridBuilder
from models.cost_weights import tempered_weights

mpl.rcParams.update({"font.family": "DejaVu Sans"})
OUT_FIG = Path(__file__).parents[1] / "outputs" / "figures"
OUT_RES = Path(__file__).parents[1] / "outputs" / "results"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(5, shuffle=True, random_state=42)

# Paper's best deep model accuracy per (dataset, embedding)
PAPER_BEST = {
    "fnfc": {
        "TF-IDF":   (89.16, "BiLSTM"),
        "Word2Vec": (86.26, "BiLSTM"),
        "GloVe":    (90.74, "GRU"),
        "BERT":     (88.31, "CNN"),
    },
    "promise": {
        "TF-IDF":   (71.13, "GRU"),
        "Word2Vec": (47.42, "LSTM"),
        "GloVe":    (79.98, "BiLSTM"),
        "BERT":     (62.37, "BiCNN"),
    },
}


def cv_acc(builder_fn, X, y, C, alpha=0.6):
    accs = []
    for tr, te in CV.split(X, y):
        fb = builder_fn(); A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
        pr = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha)).fit(A, y[tr]).predict(B)
        accs.append(accuracy_score(y[te], pr) * 100)
    return np.mean(accs)


def run_dataset(name, X, y, C, alpha):
    print(f"\n  {name.upper()} — 5-fold CV accuracy per embedding")
    results = {}
    results["TF-IDF"] = cv_acc(TFIDFBuilder, X, y, C, alpha)
    print(f"    TF-IDF:     {results['TF-IDF']:.2f}%")
    try:
        results["GloVe"] = cv_acc(GloVeBuilder, X, y, C, alpha)
        print(f"    GloVe:      {results['GloVe']:.2f}%")
    except Exception as e:
        print(f"    GloVe: skipped ({e})")
        results["GloVe"] = None
    results["Word2Vec"] = cv_acc(Word2VecBuilder, X, y, C, alpha)
    print(f"    Word2Vec:   {results['Word2Vec']:.2f}%")
    ctx_name = "sentence-transformers/all-MiniLM-L6-v2" if name == "fnfc" else "sentence-transformers/all-mpnet-base-v2"
    results["BERT"] = cv_acc(lambda: SBERTBuilder(ctx_name), X, y, C, alpha)
    print(f"    BERT/ctx:   {results['BERT']:.2f}%")
    return results


if __name__ == "__main__":
    Xf, yf = load_fnfc()
    Xp, yp = load_promise()

    res_fn = run_dataset("fnfc",    Xf, yf, C=1.5, alpha=0.6)
    res_pr = run_dataset("promise", Xp, yp, C=2.0, alpha=0.5)

    # Build comparison table
    rows = []
    for emb in ["TF-IDF", "Word2Vec", "GloVe", "BERT"]:
        paper_fn, model_fn = PAPER_BEST["fnfc"][emb]
        paper_pr, model_pr = PAPER_BEST["promise"][emb]
        rows.append({
            "Embedding":           emb,
            "FNFC_paper":          paper_fn,
            "FNFC_paper_model":    model_fn,
            "FNFC_ours":           res_fn.get(emb),
            "FNFC_gain":           (res_fn.get(emb) or 0) - paper_fn,
            "PROMISE_paper":       paper_pr,
            "PROMISE_paper_model": model_pr,
            "PROMISE_ours":        res_pr.get(emb),
            "PROMISE_gain":        (res_pr.get(emb) or 0) - paper_pr,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "embed_compare.csv", index=False)
    print("\nEmbedding comparison table:")
    print(df.to_string(index=False, float_format="%.2f"))
    print(f"\nSaved → outputs/results/embed_compare.csv")

    # Plot
    embs = ["TF-IDF", "Word2Vec", "GloVe", "BERT"]
    x = np.arange(len(embs)); w = 0.36
    PAP, OUR = "#3d5a80", "#e07a5f"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Embedding-by-embedding accuracy: paper best deep model vs. our linear classifier",
                 fontsize=12, fontweight="bold")

    for ax, ds_name, our_res in zip(axes, ["fnfc", "promise"], [res_fn, res_pr]):
        pb = [PAPER_BEST[ds_name][e][0] for e in embs]
        ou = [our_res.get(e) or 0 for e in embs]
        b1 = ax.bar(x - w/2, pb, w, color=PAP, edgecolor="white", label="Paper best (deep)")
        b2 = ax.bar(x + w/2, ou, w, color=OUR, edgecolor="white", label="Ours (LinearSVC)")
        overall_best = max(ou)
        ax.axhline(overall_best, ls="--", color="#1e7d46", lw=1.5, alpha=0.8)
        ax.text(len(embs) - 0.5, overall_best + 0.5,
                f"our overall best {overall_best:.1f}%", ha="right", fontsize=8, color="#1e7d46")
        for xi, (p, o) in enumerate(zip(pb, ou)):
            ax.text(xi - w/2, p + 0.4, f"{p:.1f}", ha="center", fontsize=8, color=PAP, fontweight="bold")
            ax.text(xi + w/2, o + 0.4, f"{o:.1f}", ha="center", fontsize=8,
                    color="#1e7d46" if o >= p else "#c0392b", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(embs, fontsize=10)
        ax.set_title(ds_name.upper(), fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if ds_name == "fnfc":
            ax.legend(fontsize=8.5, frameon=False)

    plt.tight_layout()
    out = OUT_FIG / "fig_embed_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()
