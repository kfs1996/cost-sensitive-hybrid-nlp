"""
experiments/02_benchmark.py
============================
Main benchmark: TF-IDF / hybrid + cost-sensitive LinearSVC on FNFC and PROMISE.
Reproduces Tables 2 and 3, and confirms best model vs. paper baseline.

Usage
-----
    python experiments/02_benchmark.py              # both datasets
    python experiments/02_benchmark.py --dataset fnfc
    python experiments/02_benchmark.py --dataset promise
"""

import argparse, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from data.preprocess import load_fnfc, load_promise
from models.features  import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.classifier import cross_validate, holdout_predict, print_cv_summary
from sklearn.metrics import classification_report

OUT = Path(__file__).parents[1] / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Paper baseline best results (single run, reported in Kabootari et al. 2025)
BASELINES = {
    "fnfc":    {"acc": 90.74, "model": "GRU-GloVe",    "wf1": 89.59},
    "promise": {"acc": 79.98, "model": "BiLSTM-GloVe", "wf1": 78.49},
}

CONFIGS = {
    "fnfc": {
        "loader":   load_fnfc,
        "C":        1.5,
        "alpha":    0.6,
        "n_classes": 14,
        "feature_builder": TFIDFBuilder,           # winner on FNFC
        "n_splits": 5, "n_repeats": 10,            # 50 estimates for stat tests
    },
    "promise": {
        "loader":   load_promise,
        "C":        2.0,
        "alpha":    0.5,
        "n_classes": 12,
        "feature_builder": lambda: HybridBuilder(  # winner on PROMISE
            dense_builder=SBERTBuilder("sentence-transformers/all-mpnet-base-v2")),
        "n_splits": 5, "n_repeats": 10,
    },
}


def run_dataset(name: str) -> None:
    cfg  = CONFIGS[name]
    base = BASELINES[name]
    X, y = cfg["loader"]()

    print(f"\n{'='*65}")
    print(f"Dataset : {name.upper()}  ({len(X)} reqs, {cfg['n_classes']} classes)")
    print(f"Model   : cost-sensitive LinearSVC  C={cfg['C']}  alpha={cfg['alpha']}")
    print(f"Baseline: {base['model']}  acc={base['acc']}%  wF1={base['wf1']}%")
    print(f"{'='*65}")

    # ----- Cross-validated results (50 estimates) -------------------------
    results = cross_validate(
        cfg["feature_builder"], X, y,
        C=cfg["C"], alpha=cfg["alpha"],
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"],
    )
    print_cv_summary(results, label=f"TF-IDF + SVM ({name})", baseline_acc=base["acc"])

    # Save raw CV scores
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(OUT / f"cv_scores_{name}.csv", index=False)
    print(f"  CV scores saved → outputs/results/cv_scores_{name}.csv")

    # ----- Held-out split (paper's protocol: 80/20, seed=0) ---------------
    y_test, y_pred, _ = holdout_predict(
        cfg["feature_builder"], X, y,
        C=cfg["C"], alpha=cfg["alpha"],
    )
    from sklearn.metrics import accuracy_score, f1_score
    ho_acc = accuracy_score(y_test, y_pred) * 100
    ho_mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100
    ho_wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    print(f"\n  80/20 held-out (seed=0):  acc={ho_acc:.2f}%  macroF1={ho_mf1:.2f}%  wF1={ho_wf1:.2f}%")

    # Per-class report
    report = classification_report(y_test, y_pred, zero_division=0, digits=3)
    print("\n  Per-class report (held-out):\n")
    print(report)
    with open(OUT / f"per_class_{name}.txt", "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fnfc", "promise", "both"],
                        default="both")
    args = parser.parse_args()

    datasets = ["fnfc", "promise"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds)


if __name__ == "__main__":
    main()
