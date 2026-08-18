"""
experiments/08_ablation.py
==========================
Comprehensive ablation study and hyperparameter tuning for FNFC and PROMISE.

Evaluates:
  1. Feature Ablation (TF-IDF vs. SBERT vs. Hybrid)
  2. Cost Weighting Ablation (Uniform vs. Tempered vs. Balanced)
  3. Preprocessing Ablation (Cleaned vs. Raw text)
  4. Hyperparameter Tuning (Grid search over C and alpha)

Produces:
  outputs/results/ablation_features.csv
  outputs/results/ablation_cost.csv
  outputs/results/ablation_preprocess.csv
  outputs/results/tuning_grid.csv
  outputs/figures/fig_ablation.png
  outputs/figures/fig_tuning.png
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

CV = StratifiedKFold(5, shuffle=True, random_state=42)

# Helper to run a 5-fold CV evaluation and return metrics
def evaluate_config(X, y, feature_builder_fn, C, alpha):
    accs, mf1s, wf1s = [], [], []
    for tr, te in CV.split(X, y):
        fb = feature_builder_fn()
        A = fb.fit_transform(X[tr])
        B = fb.transform(X[te])
        w = tempered_weights(y[tr], alpha)
        clf = LinearSVC(C=C, class_weight=w)
        clf.fit(A, y[tr])
        preds = clf.predict(B)
        accs.append(accuracy_score(y[te], preds) * 100)
        mf1s.append(f1_score(y[te], preds, average="macro", zero_division=0) * 100)
        wf1s.append(f1_score(y[te], preds, average="weighted", zero_division=0) * 100)
    return np.mean(accs), np.mean(mf1s), np.mean(wf1s)

def main():
    print("\n" + "="*65)
    print("  RUNNING ABLATION STUDY & HYPERPARAMETER TUNING")
    print("="*65)

    # Load datasets
    print("Loading datasets (cleaned and raw) ...")
    X_fn_clean, y_fn = load_fnfc(clean=True)
    X_fn_raw, _ = load_fnfc(clean=False)
    X_pr_clean, y_pr = load_promise(clean=True)
    X_pr_raw, _ = load_promise(clean=False)

    # Standard settings per dataset
    configs = {
        "fnfc": {
            "C": 1.5,
            "alpha": 0.6,
            "best_feat": "TFIDF",
            "tfidf_builder": lambda: TFIDFBuilder(),
            "sbert_builder": lambda: SBERTBuilder("sentence-transformers/all-MiniLM-L6-v2"),
            "hybrid_builder": lambda: HybridBuilder(dense_builder=SBERTBuilder("sentence-transformers/all-MiniLM-L6-v2")),
        },
        "promise": {
            "C": 2.0,
            "alpha": 0.5,
            "best_feat": "Hybrid",
            "tfidf_builder": lambda: TFIDFBuilder(),
            "sbert_builder": lambda: SBERTBuilder("sentence-transformers/all-mpnet-base-v2"),
            "hybrid_builder": lambda: HybridBuilder(dense_builder=SBERTBuilder("sentence-transformers/all-mpnet-base-v2")),
        }
    }

    # ---------------------------------------------------------------------------
    # 1. Feature Ablation
    # ---------------------------------------------------------------------------
    print("\n--- 1. Feature Ablation ---")
    feat_rows = []
    for ds_name, X, y in [("fnfc", X_fn_clean, y_fn), ("promise", X_pr_clean, y_pr)]:
        cfg = configs[ds_name]
        for name, builder_fn in [
            ("TF-IDF", cfg["tfidf_builder"]),
            ("SBERT", cfg["sbert_builder"]),
            ("Hybrid", cfg["hybrid_builder"])
        ]:
            acc, mf1, wf1 = evaluate_config(X, y, builder_fn, cfg["C"], cfg["alpha"])
            feat_rows.append({
                "dataset": ds_name,
                "feature": name,
                "accuracy": acc,
                "macro_f1": mf1,
                "weighted_f1": wf1
            })
            print(f"  [{ds_name.upper()}] Feature: {name:8s} -> acc={acc:.2f}%, macroF1={mf1:.2f}%, wF1={wf1:.2f}%")
    
    df_feat = pd.DataFrame(feat_rows)
    df_feat.to_csv(OUT_RES / "ablation_features.csv", index=False)

    # ---------------------------------------------------------------------------
    # 2. Cost Weighting Ablation
    # ---------------------------------------------------------------------------
    print("\n--- 2. Cost Weighting Ablation ---")
    cost_rows = []
    for ds_name, X, y in [("fnfc", X_fn_clean, y_fn), ("promise", X_pr_clean, y_pr)]:
        cfg = configs[ds_name]
        builder_fn = cfg["hybrid_builder"] if cfg["best_feat"] == "Hybrid" else cfg["tfidf_builder"]
        for weight_type, alpha_val in [
            ("Uniform (alpha=0.0)", 0.0),
            ("Tempered (alpha=opt)", cfg["alpha"]),
            ("Balanced (alpha=1.0)", 1.0)
        ]:
            acc, mf1, wf1 = evaluate_config(X, y, builder_fn, cfg["C"], alpha_val)
            cost_rows.append({
                "dataset": ds_name,
                "weighting": weight_type,
                "alpha": alpha_val,
                "accuracy": acc,
                "macro_f1": mf1,
                "weighted_f1": wf1
            })
            print(f"  [{ds_name.upper()}] Cost Weight: {weight_type:20s} -> acc={acc:.2f}%, macroF1={mf1:.2f}%, wF1={wf1:.2f}%")

    df_cost = pd.DataFrame(cost_rows)
    df_cost.to_csv(OUT_RES / "ablation_cost.csv", index=False)

    # ---------------------------------------------------------------------------
    # 3. Preprocessing Ablation
    # ---------------------------------------------------------------------------
    print("\n--- 3. Preprocessing Ablation ---")
    prep_rows = []
    for ds_name, X_clean, X_raw, y in [("fnfc", X_fn_clean, X_fn_raw, y_fn), ("promise", X_pr_clean, X_pr_raw, y_pr)]:
        cfg = configs[ds_name]
        builder_fn = cfg["hybrid_builder"] if cfg["best_feat"] == "Hybrid" else cfg["tfidf_builder"]
        for prep_type, X_data in [("Cleaned", X_clean), ("Raw text", X_raw)]:
            acc, mf1, wf1 = evaluate_config(X_data, y, builder_fn, cfg["C"], cfg["alpha"])
            prep_rows.append({
                "dataset": ds_name,
                "preprocessing": prep_type,
                "accuracy": acc,
                "macro_f1": mf1,
                "weighted_f1": wf1
            })
            print(f"  [{ds_name.upper()}] Preprocess: {prep_type:8s} -> acc={acc:.2f}%, macroF1={mf1:.2f}%, wF1={wf1:.2f}%")

    df_prep = pd.DataFrame(prep_rows)
    df_prep.to_csv(OUT_RES / "ablation_preprocess.csv", index=False)

    # ---------------------------------------------------------------------------
    # 4. Hyperparameter Grid Search
    # ---------------------------------------------------------------------------
    print("\n--- 4. Hyperparameter Grid Search (Tuning) ---")
    C_grid = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
    alpha_grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    tuning_rows = []
    
    # Run Grid search
    for ds_name, X, y in [("fnfc", X_fn_clean, y_fn), ("promise", X_pr_clean, y_pr)]:
        print(f"  Grid searching {ds_name.upper()} parameters ...")
        cfg = configs[ds_name]
        builder_fn = cfg["hybrid_builder"] if cfg["best_feat"] == "Hybrid" else cfg["tfidf_builder"]
        
        best_acc, best_mf1 = 0, 0
        best_c_acc, best_a_acc = None, None
        best_c_mf1, best_a_mf1 = None, None
        
        for C in C_grid:
            for alpha in alpha_grid:
                acc, mf1, wf1 = evaluate_config(X, y, builder_fn, C, alpha)
                tuning_rows.append({
                    "dataset": ds_name,
                    "C": C,
                    "alpha": alpha,
                    "accuracy": acc,
                    "macro_f1": mf1,
                    "weighted_f1": wf1
                })
                if acc > best_acc:
                    best_acc = acc
                    best_c_acc, best_a_acc = C, alpha
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_c_mf1, best_a_mf1 = C, alpha
                    
        print(f"    [{ds_name.upper()}] Best Accuracy: {best_acc:.2f}% (C={best_c_acc}, alpha={best_a_acc})")
        print(f"    [{ds_name.upper()}] Best Macro-F1: {best_mf1:.2f}% (C={best_c_mf1}, alpha={best_a_mf1})")

    df_tuning = pd.DataFrame(tuning_rows)
    df_tuning.to_csv(OUT_RES / "tuning_grid.csv", index=False)

    # ---------------------------------------------------------------------------
    # Plotting Ablation Studies
    # ---------------------------------------------------------------------------
    print("\nGenerating ablation plots ...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Ablation Study Results on FNFC and PROMISE", fontsize=13, fontweight="bold", y=0.98)
    
    # 1. Feature plot
    ax = axes[0]
    x = np.arange(3)
    w = 0.25
    fn_sub = df_feat[df_feat.dataset == "fnfc"]
    pr_sub = df_feat[df_feat.dataset == "promise"]
    ax.bar(x - w/2, fn_sub.accuracy, w, label="FNFC Accuracy", color="#3d5a80", edgecolor="white")
    ax.bar(x + w/2, pr_sub.accuracy, w, label="PROMISE Accuracy", color="#e07a5f", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(fn_sub.feature, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 95)
    ax.set_title("A · Feature Representations", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    
    # 2. Cost plot
    ax = axes[1]
    x = np.arange(3)
    fn_sub = df_cost[df_cost.dataset == "fnfc"]
    pr_sub = df_cost[df_cost.dataset == "promise"]
    ax.bar(x - w/2, fn_sub.macro_f1, w, label="FNFC Macro-F1", color="#3d5a80", edgecolor="white")
    ax.bar(x + w/2, pr_sub.macro_f1, w, label="PROMISE Macro-F1", color="#e07a5f", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Uniform", "Tempered", "Balanced"], fontsize=9)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(25, 65)
    ax.set_title("B · Cost weighting scheme (alpha)", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    
    # 3. Preprocess plot
    ax = axes[2]
    x = np.arange(2)
    fn_sub = df_prep[df_prep.dataset == "fnfc"]
    pr_sub = df_prep[df_prep.dataset == "promise"]
    ax.bar(x - w/2, fn_sub.accuracy, w, label="FNFC Accuracy", color="#3d5a80", edgecolor="white")
    ax.bar(x + w/2, pr_sub.accuracy, w, label="PROMISE Accuracy", color="#e07a5f", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Cleaned Text", "Raw Text"], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(65, 95)
    ax.set_title("C · Text Preprocessing", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_ablation.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved ablation plot -> outputs/figures/fig_ablation.png")

    # Heatmaps for Grid Search
    print("Generating grid search heatmaps ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Hyperparameter Tuning: Accuracy Heatmaps", fontsize=13, fontweight="bold")
    
    for i, ds_name in enumerate(["fnfc", "promise"]):
        ax = axes[i]
        sub = df_tuning[df_tuning.dataset == ds_name]
        pivot = sub.pivot(index="C", columns="alpha", values="accuracy")
        
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(alpha_grid)))
        ax.set_xticklabels([f"{a:.1f}" for a in alpha_grid])
        ax.set_yticks(range(len(C_grid)))
        ax.set_yticklabels([f"{c:.1f}" for c in C_grid])
        
        # Add labels inside cells
        for y_idx in range(len(C_grid)):
            for x_idx in range(len(alpha_grid)):
                val = pivot.values[y_idx, x_idx]
                ax.text(x_idx, y_idx, f"{val:.1f}", ha="center", va="center", 
                        color="black" if val < pivot.values.max() - 5 else "white", fontsize=8.5, fontweight="bold")
                
        ax.set_xlabel("Alpha (cost-sensitivity exponent)")
        ax.set_ylabel("C (regularisation strength)")
        ax.set_title(f"{ds_name.upper()} (Optimal: C={pivot.index[np.argmax(pivot.values) // len(alpha_grid)]}, alpha={pivot.columns[np.argmax(pivot.values) % len(alpha_grid)]})", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_tuning.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved tuning plot -> outputs/figures/fig_tuning.png")
    
    print("\n" + "="*65)
    print("  ABLATION STUDY & HYPERPARAMETER TUNING COMPLETED")
    print("="*65)

if __name__ == "__main__":
    main()
