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
from models.features import TFIDFBuilder, SEBERTBuilder, REBERTBuilder
from models.ensemble import cross_validate_ensemble

def main():
    datasets = {
        "FNFC": load_fnfc,
        "PROMISE": load_promise
    }
    
    results = []
    
    # We will test two custom base models
    base_configurations = {
        "SEBERT_Hybrid": [("tfidf", TFIDFBuilder, "svm"), ("sebert", SEBERTBuilder, "svm")],
        "REBERT_Hybrid": [("tfidf", TFIDFBuilder, "svm"), ("rebert", REBERTBuilder, "svm")]
    }
    
    # Standard alpha for all tests
    alpha = 0.5
    
    for d_name, load_fn in datasets.items():
        print(f"\n=========================================")
        print(f"Running Domain BERT Benchmark on: {d_name}")
        print(f"=========================================")
        
        X, y = load_fn()
        
        for config_name, base_models in base_configurations.items():
            for stacking_mode in ["hard", "soft"]:
                for meta_clf in ["logreg", "rf", "svm"]:
                    print(f"Config: {config_name} | Stacking: {stacking_mode} | Meta: {meta_clf}")
                    
                    start_time = time.time()
                    metrics = cross_validate_ensemble(
                        X, y,
                        base_models=base_models,
                        meta_classifier=meta_clf,
                        alpha=alpha,
                        stacking_mode=stacking_mode,
                        tune_meta=False,  # Turn off grid search to save some time
                        n_splits=5
                    )
                    elapsed = time.time() - start_time
                    
                    acc = metrics['accuracy'].mean()
                    macro_f1 = metrics['macro_f1'].mean()
                    weighted_f1 = metrics['weighted_f1'].mean()
                    
                    results.append({
                        "Dataset": d_name,
                        "Configuration": config_name,
                        "Meta_Clf": meta_clf,
                        "Stacking_Mode": stacking_mode,
                        "Alpha": alpha,
                        "Accuracy": acc,
                        "Macro_F1": macro_f1,
                        "Weighted_F1": weighted_f1,
                        "Time_sec": elapsed
                    })
                    
                    # Save incremental progress
                    df_results = pd.DataFrame(results)
                    out_dir = Path("outputs/results")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "11_domain_bert_benchmark.csv"
                    df_results.to_csv(out_path, index=False)
                    
    print("\nBenchmark Done! Results saved to outputs/results/11_domain_bert_benchmark.csv")

if __name__ == "__main__":
    main()
