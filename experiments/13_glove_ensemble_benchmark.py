# experiments/13_glove_ensemble_benchmark.py
"""
Benchmark script for the Triple‑Hybrid Stacking Ensemble (TF‑IDF + SBERT + GloVe).
Runs the same evaluation as 09_master_benchmark.py but adds GloVe as a separate base model.
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from pathlib import Path

# Ensure single‑threaded environment on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder, GloVeBuilder
from models.ensemble import cross_validate_ensemble


def get_hybrid_builder():
    # Keep the original hybrid (TF‑IDF + SBERT) for comparison
    from models.features import HybridBuilder
    return HybridBuilder(dense_builder=SBERTBuilder())

# Phase‑1 single models (unchanged)
SINGLE_MODELS = [
    ("TFIDF", TFIDFBuilder),
    ("BERT", SBERTBuilder),
    ("GloVe", GloVeBuilder),
    ("Hybrid", get_hybrid_builder),
]

# Phase‑2 stacking base experts – now three distinct experts
STACKING_BASE = [
    ("TFIDF", TFIDFBuilder, "svm"),
    ("BERT", SBERTBuilder, "svm"),
    ("GloVe", GloVeBuilder, "svm"),
]

META_CLFS = ["logreg", "rf", "svm"]
ALPHAS = [0.0, 0.5]
STACKING_MODES = ["hard", "soft"]
DATASETS = {"FNFC": load_fnfc, "PROMISE": load_promise}


def main():
    results = []
    out_dir = Path(__file__).parents[2] / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "glove_ensemble_benchmark.csv"
    print(f"Saving results to: {out_file}")

    for ds_name, load_fn in DATASETS.items():
        print(f"\n=========================================")
        print(f"Running dataset: {ds_name}")
        print(f"=========================================")
        X, y = load_fn()
        # Phase‑1 single models – include GloVe now
        for model_name, builder in SINGLE_MODELS:
            for alpha in ALPHAS:
                C = 2.0 if ds_name == "PROMISE" else 1.5
                # Use CostSensitiveSVM directly for single‑model evaluation
                from models.classifier import CostSensitiveSVM, cross_validate as cv_single
                print(f"Phase 1 | {model_name} | alpha={alpha}")
                start = time.time()
                res = cv_single(builder, X, y, C=C, alpha=alpha, n_splits=5)
                end = time.time()
                results.append({
                    "Phase": "1_Single",
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Meta_Clf": "N/A",
                    "Stacking_Mode": "N/A",
                    "Alpha": alpha,
                    "Accuracy": np.mean(res["accuracy"]),
                    "Macro_F1": np.mean(res["macro_f1"]),
                    "Weighted_F1": np.mean(res["weighted_f1"]),
                    "Time_sec": end - start,
                })
                pd.DataFrame(results).to_csv(out_file, index=False)
        # Phase‑2 stacking models
        for mode in STACKING_MODES:
            for meta in META_CLFS:
                for alpha in ALPHAS:
                    print(f"Phase 2 | Stacking: {mode} | Meta: {meta} | alpha={alpha}")
                    start = time.time()
                    res = cross_validate_ensemble(
                        X, y,
                        base_models=STACKING_BASE,
                        meta_classifier=meta,
                        alpha=alpha,
                        stacking_mode=mode,
                        n_splits=5,
                    )
                    end = time.time()
                    results.append({
                        "Phase": f"2_Stacking_{mode}",
                        "Dataset": ds_name,
                        "Model": "Lexical_Semantic_GloVe",
                        "Meta_Clf": meta,
                        "Stacking_Mode": mode,
                        "Alpha": alpha,
                        "Accuracy": np.mean(res["accuracy"]),
                        "Macro_F1": np.mean(res["macro_f1"]),
                        "Weighted_F1": np.mean(res["weighted_f1"]),
                        "Time_sec": end - start,
                    })
                    pd.DataFrame(results).to_csv(out_file, index=False)

if __name__ == "__main__":
    main()
