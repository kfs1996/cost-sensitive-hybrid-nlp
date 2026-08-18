import pandas as pd
import numpy as np
import time
import os
import sys

# Prevent PyTorch/Transformers thread locking on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.classifier import CostSensitiveSVM, cross_validate as cv_single
from models.ensemble import cross_validate_ensemble

def get_hybrid_builder():
    return HybridBuilder(dense_builder=SBERTBuilder())

# Phase 1
SINGLE_MODELS = [
    ("TFIDF", TFIDFBuilder),
    ("BERT", SBERTBuilder),
    ("Hybrid", get_hybrid_builder)
]

# Phase 2
STACKING_BASE = [
    ("TFIDF", TFIDFBuilder, "svm"),
    ("BERT", SBERTBuilder, "svm"),
    ("Hybrid", get_hybrid_builder, "svm")
]

META_CLFS = ["logreg", "rf", "svm"]
ALPHAS = [0.0, 0.5]
STACKING_MODES = ["hard", "soft"]
DATASETS = {"FNFC": load_fnfc, "PROMISE": load_promise}

def main():
    results = []
    out_dir = Path(__file__).parents[1] / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "master_benchmark.csv"
    
    print(f"Saving results to: {out_file}")
    
    for ds_name, load_fn in DATASETS.items():
        print(f"\n=========================================")
        print(f"Running dataset: {ds_name}")
        print(f"=========================================")
        X, y = load_fn()
        
        # 1. PHASE 1: Single Models
        for model_name, builder in SINGLE_MODELS:
            for alpha in ALPHAS:
                C = 2.0 if ds_name == "PROMISE" else 1.5
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
                    "Accuracy": np.mean(res['accuracy']),
                    "Macro_F1": np.mean(res['macro_f1']),
                    "Weighted_F1": np.mean(res['weighted_f1']),
                    "Time_sec": end - start
                })
                df = pd.DataFrame(results)
                df.to_csv(out_file, index=False)
                
        # 2. PHASE 2: Stacking Models
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
                        n_splits=5
                    )
                    end = time.time()
                    results.append({
                        "Phase": f"2_Stacking_{mode}",
                        "Dataset": ds_name,
                        "Model": "Lexical_Semantic_Hybrid",
                        "Meta_Clf": meta,
                        "Stacking_Mode": mode,
                        "Alpha": alpha,
                        "Accuracy": np.mean(res['accuracy']),
                        "Macro_F1": np.mean(res['macro_f1']),
                        "Weighted_F1": np.mean(res['weighted_f1']),
                        "Time_sec": end - start
                    })
                    df = pd.DataFrame(results)
                    df.to_csv(out_file, index=False)

if __name__ == "__main__":
    main()
