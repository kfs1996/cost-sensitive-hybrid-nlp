"""
experiments/08_ensemble_benchmark.py
====================================
Runs the Phase 2 Stacking Ensemble grid (36 combinations).
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.ensemble import cross_validate_ensemble
from models.classifier import print_cv_summary

OUT = Path(__file__).parents[1] / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

def get_hybrid_builder():
    return HybridBuilder(dense_builder=SBERTBuilder())

TEAMS = {
    "Lexical_Semantic_Hybrid": [
        ("TFIDF", TFIDFBuilder, "svm"),
        ("BERT", SBERTBuilder, "svm"),
        ("Hybrid", get_hybrid_builder, "svm")
    ]
}

META_CLASSIFIERS = ["logreg"]
COST_AWARENESS = [0.0, 0.5] # 0.0 = Standard, 0.5 = Cost-Aware

def run_single_combination(dataset_name, X, y, team_name, base_models, meta, alpha):
    name = f"{team_name} | {meta} | alpha={alpha}"
    print(f"Starting: {name}")
    metrics = cross_validate_ensemble(
        X, y,
        base_models=base_models,
        meta_classifier=meta,
        alpha=alpha,
        n_splits=5 # 5-fold CV
    )
    print_cv_summary(metrics, label=name)
    return {
        "dataset": dataset_name,
        "team": team_name,
        "meta_clf": meta,
        "alpha": alpha,
        "accuracy": metrics["accuracy"].mean(),
        "macro_f1": metrics["macro_f1"].mean(),
        "weighted_f1": metrics["weighted_f1"].mean()
    }

def run_grid(dataset_name, X, y):
    print(f"\n{'='*65}")
    print(f"--- Running Phase 2 Grid on {dataset_name.upper()} SEQUENTIALLY ---")
    print(f"{'='*65}")
    
    tasks = []
    for team_name, base_models in TEAMS.items():
        for meta in META_CLASSIFIERS:
            for alpha in COST_AWARENESS:
                tasks.append((team_name, base_models, meta, alpha))
                
    results = []
    for t, b, m, a in tasks:
        res = run_single_combination(dataset_name, X, y, t, b, m, a)
        results.append(res)
                
    df = pd.DataFrame(results)
    out_path = OUT / f"ensemble_results_{dataset_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fnfc", "promise", "both"], default="both")
    args = parser.parse_args()
    
    if args.dataset in ["fnfc", "both"]:
        X_f, y_f = load_fnfc()
        run_grid("fnfc", X_f, y_f)
        
    if args.dataset in ["promise", "both"]:
        X_p, y_p = load_promise()
        run_grid("promise", X_p, y_p)
