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
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder
from models.ensemble import cross_validate_ensemble

def main():
    datasets = {
        "FNFC": load_fnfc,
        "PROMISE": load_promise
    }
    
    results = []
    
    # We only test the Hybrid base_models (TF-IDF + BERT)
    hybrid_base_models = [
        ("tfidf", TFIDFBuilder, "svm"),
        ("bert", SBERTBuilder, "svm")
    ]
    
    # We only care about Phase 2 Cost-Aware alpha=0.5
    alpha = 0.5
    
    for d_name, load_fn in datasets.items():
        print(f"\n=========================================")
        print(f"Running Grid Search on dataset: {d_name}")
        print(f"=========================================")
        
        X, y = load_fn()
        
        for stacking_mode in ["hard", "soft"]:
            for meta_clf in ["logreg", "rf", "svm"]:
                print(f"Phase 2 | Stacking: {stacking_mode} | Meta: {meta_clf} | alpha={alpha} | GridSearch=True")
                
                start_time = time.time()
                metrics = cross_validate_ensemble(
                    X, y,
                    base_models=hybrid_base_models,
                    meta_classifier=meta_clf,
                    alpha=alpha,
                    stacking_mode=stacking_mode,
                    tune_meta=True,  # Enable Grid Search!
                    n_splits=5
                )
                elapsed = time.time() - start_time
                
                acc = metrics['accuracy'].mean()
                macro_f1 = metrics['macro_f1'].mean()
                weighted_f1 = metrics['weighted_f1'].mean()
                
                results.append({
                    "Dataset": d_name,
                    "Meta_Clf": meta_clf,
                    "Stacking_Mode": stacking_mode,
                    "Alpha": alpha,
                    "Accuracy": acc,
                    "Macro_F1": macro_f1,
                    "Weighted_F1": weighted_f1,
                    "Time_sec": elapsed
                })
            
    df_results = pd.DataFrame(results)
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "10_grid_search_benchmark.csv"
    print(f"\nSaving results to: {out_path}")
    df_results.to_csv(out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
