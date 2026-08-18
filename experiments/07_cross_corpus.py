"""
experiments/07_cross_corpus.py
================================
Cross-corpus generalisation study (Section 5.6, Table 5 in the paper).

Trains on one corpus and tests on the other — in both directions — using the
harmonised 11-class taxonomy. Compares TF-IDF vs. contextual embeddings,
and uniform vs. cost-sensitive weighting.

Produces
--------
  outputs/results/cross_corpus.csv
  outputs/figures/fig_cross_corpus.png
"""

import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    _orig_send = requests.Session.send
    def _patched_send(self, request, **kwargs):
        kwargs['verify'] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _patched_send
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from data.preprocess import load_harmonised, SHARED_CLASSES
from models.features  import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.cost_weights import tempered_weights

mpl.rcParams.update({"font.family": "DejaVu Sans"})
OUT_FIG = Path(__file__).parents[1] / "outputs" / "figures"
OUT_RES = Path(__file__).parents[1] / "outputs" / "results"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

# Shared sentence-transformer model (same space for both corpora)
SBERT_MODEL = "all-MiniLM-L6-v2"

# In-domain ceilings for reference (5-fold CV on individual full datasets)
IN_DOMAIN = {
    "FNFC":    {"acc": 92.9, "macro_f1": 45.5},
    "PROMISE": {"acc": 80.7, "macro_f1": 64.4},
}


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "macro_f1":  f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
    }


def transfer(X_tr, y_tr, X_te, y_te, feat: str, alpha: float,
             C: float = 1.0, sbert_emb_tr=None, sbert_emb_te=None) -> dict:
    """
    Train on (X_tr, y_tr) and test on (X_te, y_te).

    feat  : 'tfidf' | 'contextual'
    alpha : cost-sensitivity exponent
    """
    if feat == "tfidf":
        fb = TFIDFBuilder(); A = fb.fit_transform(X_tr); B = fb.transform(X_te)
    else:                               # contextual: pre-computed embeddings
        A = sbert_emb_tr; B = sbert_emb_te

    clf = LinearSVC(C=C, class_weight=tempered_weights(y_tr, alpha))
    clf.fit(A, y_tr)
    return metrics(y_te, clf.predict(B))


def indomain_cv(X, y, feat: str, alpha: float, C: float = 1.0,
                sbert_emb=None) -> dict:
    """5-fold CV ceiling on the same corpus."""
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    acc, mf1 = [], []
    for tr, te in cv.split(X, y):
        if feat == "tfidf":
            fb = TFIDFBuilder(); A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
        else:
            A = sbert_emb[tr]; B = sbert_emb[te]
        clf = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha)).fit(A, y[tr])
        pr  = clf.predict(B)
        acc.append(accuracy_score(y[te], pr) * 100)
        mf1.append(f1_score(y[te], pr, average="macro", zero_division=0) * 100)
    return {"accuracy": np.mean(acc), "macro_f1": np.mean(mf1)}


if __name__ == "__main__":
    print("Loading harmonised datasets …")
    X_fn, y_fn, X_pr, y_pr = load_harmonised()
    print(f"  FNFC   : {len(X_fn)} rows,  F-share {(y_fn=='F').mean()*100:.1f}%")
    print(f"  PROMISE: {len(X_pr)} rows,  F-share {(y_pr=='F').mean()*100:.1f}%")

    # Pre-compute shared-space sentence embeddings (one model, both corpora)
    print(f"\nEncoding with {SBERT_MODEL} …")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(f"sentence-transformers/{SBERT_MODEL}", device="cpu")
    E_fn = sbert.encode(list(X_fn), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    E_pr = sbert.encode(list(X_pr), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    print(f"  FNFC emb: {E_fn.shape}   PROMISE emb: {E_pr.shape}")

    # ---------------------------------------------------------------------------
    # Cross-corpus transfer: all 4 configurations × 2 directions
    # ---------------------------------------------------------------------------
    configs = [
        ("tfidf",      0.0, "TF-IDF, uniform"),
        ("tfidf",      1.0, "TF-IDF, cost-sensitive"),
        ("contextual", 0.0, "Contextual, uniform"),
        ("contextual", 1.0, "Contextual, cost-sensitive"),
    ]

    rows = []
    for direction, (X_tr, y_tr, E_tr), (X_te, y_te, E_te), src, tgt in [
        ("FNFC → PROMISE", (X_fn, y_fn, E_fn), (X_pr, y_pr, E_pr), "FNFC",    "PROMISE"),
        ("PROMISE → FNFC", (X_pr, y_pr, E_pr), (X_fn, y_fn, E_fn), "PROMISE", "FNFC"),
    ]:
        print(f"\n{'—'*55}")
        print(f"  Direction: {direction}   (test F-share {(y_te=='F').mean()*100:.0f}%)")
        print(f"{'—'*55}")
        for feat, alpha, label in configs:
            m = transfer(X_tr, y_tr, X_te, y_te, feat=feat, alpha=alpha,
                         sbert_emb_tr=E_tr, sbert_emb_te=E_te)
            rows.append({
                "direction": direction, "feat": feat, "alpha": alpha,
                "label": label, "src": src, "tgt": tgt,
                **m,
            })
            print(f"    {label:30s}  acc={m['accuracy']:.1f}  macroF1={m['macro_f1']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "cross_corpus.csv", index=False)
    print(f"\nSaved → outputs/results/cross_corpus.csv")

    # ---------------------------------------------------------------------------
    # In-domain ceilings (for the plot reference lines)
    # ---------------------------------------------------------------------------
    print("\n  In-domain ceilings (5-fold CV, TF-IDF, uniform) …")
    c_fn = indomain_cv(X_fn, y_fn, "tfidf", 0.0)
    c_pr = indomain_cv(X_pr, y_pr, "tfidf", 0.0)
    ceilings = {"FNFC": c_fn, "PROMISE": c_pr}
    print(f"    FNFC:    acc={c_fn['accuracy']:.1f}%  macroF1={c_fn['macro_f1']:.1f}%")
    print(f"    PROMISE: acc={c_pr['accuracy']:.1f}%  macroF1={c_pr['macro_f1']:.1f}%")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    ACC_COL, MF1_COL = "#3d5a80", "#e07a5f"
    directions_list = ["FNFC → PROMISE", "PROMISE → FNFC"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Cross-corpus generalisation — harmonised 11-class taxonomy",
                 fontsize=13, fontweight="bold", y=0.97)

    for ax, direction in zip(axes, directions_list):
        sub = df[df.direction == direction].reset_index(drop=True)
        tgt = sub.iloc[0]["tgt"]
        ceil = ceilings[tgt]
        x = np.arange(len(sub)); w = 0.38
        b1 = ax.bar(x - w/2, sub.accuracy,  w, color=ACC_COL, edgecolor="white", label="Accuracy")
        b2 = ax.bar(x + w/2, sub.macro_f1,  w, color=MF1_COL, edgecolor="white", label="Macro-F1")
        ax.axhline(ceil["accuracy"],  ls="--", color=ACC_COL, lw=1.3, alpha=0.7)
        ax.axhline(ceil["macro_f1"], ls=":",  color=MF1_COL, lw=1.3, alpha=0.8)
        ax.text(len(sub) - 0.5, ceil["accuracy"] + 0.5,
                f"acc ceiling {ceil['accuracy']:.0f}", ha="right", fontsize=7.5, color=ACC_COL)
        ax.text(-0.45, ceil["macro_f1"] + 0.5,
                f"mF1 ceiling {ceil['macro_f1']:.0f}", fontsize=7.5, color=MF1_COL)
        for xi, (a, m) in enumerate(zip(sub.accuracy, sub.macro_f1)):
            ax.text(xi - w/2, a + 0.4, f"{a:.0f}", ha="center", fontsize=8, color=ACC_COL, fontweight="bold")
            ax.text(xi + w/2, m + 0.4, f"{m:.0f}", ha="center", fontsize=8, color=MF1_COL, fontweight="bold")
        best_idx = sub.accuracy.idxmax()
        ax.add_patch(plt.Rectangle((best_idx - w, sub.accuracy.min() - 1),
                                   2 * w, sub.accuracy[best_idx] + 1 - sub.accuracy.min() + 1,
                                   fill=False, edgecolor="#1e7d46", lw=2, zorder=5))
        ax.set_xticks(x)
        ax.set_xticklabels(sub.label.str.replace(", ", "\n"), fontsize=8)
        ax.set_title(f"{direction} (test F-share {'50' if 'PROMISE' in direction.split('→')[1] else '88'}%)",
                     fontsize=11, fontweight="bold", loc="left", pad=6)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.set_ylim(0, max(sub.accuracy.max(), ceil["accuracy"]) + 10)
        ax.spines[["top", "right"]].set_visible(False)
        if direction == directions_list[0]:
            ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.text(0.5, 0.005,
             "Green box = best config per direction. Dashed/dotted lines = in-domain ceilings. "
             "Contextual embeddings transfer better; cost-sensitivity helps imbalanced→balanced only.",
             ha="center", fontsize=7.8, color="#555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUT_FIG / "fig_cross_corpus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()
