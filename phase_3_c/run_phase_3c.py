import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import itertools
import os
from joblib import Parallel, delayed

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data.preprocess import load_fnfc, load_promise
from models.deep_embeddings import get_deep_embeddings
from models.cost_weights import tempered_weights

ALGORITHMS = [
    'CS-SVM (Linear)', 'CS-SVM (RBF)', 'CS-DT', 'CS-LR', 'CS-RF'
]

ALL_EMBEDDINGS = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'SBERT', 'MPNet']
HYBRID_COMBOS = list(itertools.combinations(ALL_EMBEDDINGS, 3))

def get_ml_model(algo, w_dict):
    if algo == 'CS-SVM (Linear)':
        return LinearSVC(class_weight=w_dict, random_state=42)
    elif algo == 'CS-SVM (RBF)':
        return SVC(kernel='rbf', class_weight=w_dict, random_state=42)
    elif algo == 'CS-DT':
        return DecisionTreeClassifier(class_weight=w_dict, random_state=42)
    elif algo == 'CS-LR':
        return LogisticRegression(class_weight=w_dict, random_state=42, max_iter=1000)
    elif algo == 'CS-RF':
        return RandomForestClassifier(class_weight=w_dict, n_estimators=100, random_state=42, n_jobs=1) # force RF to use 1 thread internally since we parallelize externally

def pool_to_2d(emb):
    if hasattr(emb, "toarray"):
        emb = emb.toarray()
    if len(emb.shape) == 3:
        return emb.mean(axis=1)
    return emb

def evaluate_model(ds_name, algo, combo_name, fold_idx, X_train_hyb, X_test_hyb, y_train, y_test, w_dict, out_file):
    model = get_ml_model(algo, w_dict)
    model.fit(X_train_hyb, y_train)
    preds = model.predict(X_test_hyb)
    acc = accuracy_score(y_test, preds)
    
    with open(out_file, "a") as f:
        f.write(f"{ds_name},{algo},{combo_name},{fold_idx},{acc}\n")
    return acc

def main():
    print(f"Loading data for Phase 3-C (Tri-Way Hybrid Cost-Sensitive Classical ML)...")
    print(f"Maximum Parallel Cores Unlocked: {os.cpu_count()}")
    
    X_fnfc, y_fnfc = load_fnfc(clean=True)
    X_prom, y_prom = load_promise(clean=True)
    
    datasets = {
        'FNFC': (X_fnfc, y_fnfc),
        'PROMISE': (X_prom, y_prom)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    out_dir = _ROOT / "phase_3_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tri_hybrid_csl_ml_reproduction.csv"
    
    if not out_file.exists():
        with open(out_file, "w") as f:
            f.write("dataset,algo,embed,fold,acc\n")
            
    print(f"Total Tri-Way Combinations: {len(HYBRID_COMBOS)}")
    
    for ds_name, (X, y) in datasets.items():
        print(f"\nProcessing {ds_name}...")
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            w_dict = tempered_weights(y_train, alpha=0.5)
            
            base_embs = {}
            for base_name in ALL_EMBEDDINGS:
                print(f"  Extracting {base_name}...", end="", flush=True)
                t0 = time.time()
                train_emb, test_emb, _ = get_deep_embeddings(X_train, X_test, base_name)
                base_embs[base_name] = (pool_to_2d(train_emb), pool_to_2d(test_emb))
                print(f" done ({time.time()-t0:.1f}s)")
            
            # Prepare all parallel jobs for this fold
            tasks = []
            print(f"  Launching {len(HYBRID_COMBOS) * len(ALGORITHMS)} models simultaneously across all CPU cores...")
            for combo in HYBRID_COMBOS:
                combo_name = "+".join(combo)
                train_parts = [base_embs[name][0] for name in combo]
                test_parts = [base_embs[name][1] for name in combo]
                
                X_train_hyb = np.hstack(train_parts).astype(np.float32)
                X_test_hyb = np.hstack(test_parts).astype(np.float32)
                
                for algo in ALGORITHMS:
                    tasks.append((ds_name, algo, combo_name, fold_idx, X_train_hyb, X_test_hyb, y_train, y_test, w_dict, out_file))
            
            # Unleash max CPU parallelization
            Parallel(n_jobs=-1, backend="loky")(
                delayed(evaluate_model)(*t) for t in tasks
            )
            print(f"  Fold {fold_idx} completed instantly!")

if __name__ == "__main__":
    main()
