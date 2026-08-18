# experiments/14_glove_phase2_2_1_2_2.py
"""
Run Phase 2 stacking experiments (hard and soft modes) with the Triple‑Hybrid base set
including GloVe, using a Logistic Regression meta‑classifier. This captures the
requested 2.1 (hard) and 2.2 (soft) configurations and writes the results to CSV.
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from pathlib import Path

# Single‑threaded settings for Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder, GloVeBuilder
from models.ensemble import cross_validate_ensemble

# Base experts for Phase 2 (TF‑IDF, BERT, GloVe)
STACKING_BASE = [
    ("TFIDF", TFIDFBuilder, "svm"),
    ("BERT", SBERTBuilder, "svm"),
    ("GloVe", GloVeBuilder, "svm"),
]

META_CLF = "logreg"  # Logistic Regression meta‑classifier
ALPHA = 0.5          # Use the cost‑weighting exponent that gave the best results in Phase 1
STACKING_MODES = ["hard", "soft"]  # 2.1 = hard, 2.2 = soft
DATASETS = {"FNFC": load_fnfc, "PROMISE": load_promise}

def main():
    results = []
    out_dir = Path(__file__).parents[2] / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "glove_phase2_hard_soft.csv"
    print(f"Saving Phase 2 results to: {out_file}")

    for ds_name, load_fn in DATASETS.items():
        print(f"\n=== Running Phase 2 on {ds_name} ===")
        X, y = load_fn()
        for mode in STACKING_MODES:
            print(f"Phase 2 | mode={mode}")
            start = time.time()
            res = cross_validate_ensemble(
                X, y,
                base_models=STACKING_BASE,
                meta_classifier=META_CLF,
                alpha=ALPHA,
                stacking_mode=mode,
                n_splits=5,
            )
            end = time.time()
            results.append({
                "Phase": f"2_Stacking_{mode}",
                "Dataset": ds_name,
                "Model": "Lexical_Semantic_GloVe",
                "Meta_Clf": META_CLF,
                "Stacking_Mode": mode,
                "Alpha": ALPHA,
                "Accuracy": np.mean(res["accuracy"]),
                "Macro_F1": np.mean(res["macro_f1"]),
                "Weighted_F1": np.mean(res["weighted_f1"]),
                "Time_sec": end - start,
            })
            pd.DataFrame(results).to_csv(out_file, index=False)

if __name__ == "__main__":
    main()


