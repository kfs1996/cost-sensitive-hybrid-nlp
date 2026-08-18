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
from concurrent.futures import ProcessPoolExecutor
import os

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data.preprocess import load_fnfc, load_promise
from models.deep_embeddings import get_deep_embeddings
from models.cost_weights import tempered_weights
from models.custom_csl import MetaCost, AdaCost, CSKNN

# 8 Pure CSL Algorithms (Native + Custom)
ALGORITHMS = [
    'CS-SVM (Linear)', 'CS-SVM (RBF)', 'CS-DT', 'CS-LR', 'CS-RF',
    'MetaCost', 'AdaCost', 'CS-KNN'
]
EMBEDDINGS = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'SBERT', 'MPNet']

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
        return RandomForestClassifier(class_weight=w_dict, random_state=42)
    elif algo == 'MetaCost':
        return MetaCost(cost_matrix=w_dict, n_estimators=10)
    elif algo == 'AdaCost':
        return AdaCost(cost_matrix=w_dict, n_estimators=50)
    elif algo == 'CS-KNN':
        return CSKNN(cost_matrix=w_dict, n_neighbors=5)

def train_and_evaluate_csl_ml(dataset_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes):
    if len(X_train_emb.shape) == 3:
        X_train_emb = X_train_emb.mean(axis=1)
        X_test_emb = X_test_emb.mean(axis=1)
        
    w_dict = tempered_weights(y_train, alpha=0.5)
    
    model = get_ml_model(algo, w_dict)
    model.fit(X_train_emb, y_train)
    
    preds = model.predict(X_test_emb)
    acc = accuracy_score(y_test, preds)
    print(f"[{dataset_name}] {algo} + {embed} | Fold {fold_idx} | Acc: {acc:.4f}")
    return dataset_name, algo, embed, fold_idx, acc

def main():
    print("Loading data for Phase 2-B (8 Pure CSL Algorithms)...")
    X_fnfc, y_fnfc = load_fnfc(clean=True)
    X_prom, y_prom = load_promise(clean=True)
    
    datasets = {
        'FNFC': (X_fnfc, y_fnfc, len(np.unique(y_fnfc))),
        'PROMISE': (X_prom, y_prom, len(np.unique(y_prom)))
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tasks = []
    
    for ds_name, (X, y, num_classes) in datasets.items():
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for embed in EMBEDDINGS:
                X_train_emb, X_test_emb, _ = get_deep_embeddings(X_train, X_test, embed)
                
                if hasattr(X_train_emb, "toarray"):
                    X_train_emb = X_train_emb.toarray()
                    X_test_emb = X_test_emb.toarray()
                    
                for algo in ALGORITHMS:
                    tasks.append((ds_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes))
                    
    print(f"Submitting {len(tasks)} jobs...")
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(train_and_evaluate_csl_ml, *t) for t in tasks]
        for f in futures:
            results.append(f.result())
            
    df = pd.DataFrame(results, columns=['dataset', 'algo', 'embed', 'fold', 'acc'])
    summary = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()
    summary['config'] = summary['algo'] + '-' + summary['embed']
    
    out_dir = _ROOT / "phase_2_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "csl_ml_baseline_reproduction.csv", index=False)
    print("Saved Phase 2-B (8 Pure CSL)!")

if __name__ == "__main__":
    main()
