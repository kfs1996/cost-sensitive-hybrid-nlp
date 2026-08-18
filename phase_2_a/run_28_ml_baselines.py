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

from utils.plot_metrics import generate_and_save_plots

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from data.preprocess import load_fnfc, load_promise
from models.deep_embeddings import get_deep_embeddings

ALGORITHMS = ['LinearSVM', 'SVM-RBF', 'DecisionTree', 'LogisticRegression', 'RandomForest', 'AdaBoost', 'NaiveBayes']
EMBEDDINGS = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'MPNet']

def get_ml_model(algo):
    if algo == 'LinearSVM':
        return LinearSVC(random_state=42)
    elif algo == 'SVM-RBF':
        return SVC(kernel='rbf', probability=True, random_state=42)
    elif algo == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif algo == 'LogisticRegression':
        return LogisticRegression(random_state=42, max_iter=1000)
    elif algo == 'RandomForest':
        return RandomForestClassifier(random_state=42)
    elif algo == 'AdaBoost':
        return AdaBoostClassifier(random_state=42)
    elif algo == 'NaiveBayes':
        return GaussianNB()

def train_and_evaluate_ml(dataset_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes):
    # Flatten 3D to 2D
    if len(X_train_emb.shape) == 3:
        X_train_emb = X_train_emb.mean(axis=1)
        X_test_emb = X_test_emb.mean(axis=1)
        
    model = get_ml_model(algo)
    model.fit(X_train_emb, y_train)
    preds = model.predict(X_test_emb)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_emb)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test_emb)
    else:
        y_prob = None
        
    generate_and_save_plots(y_test, preds, y_prob, num_classes, "Phase_2-A", dataset_name, embed, algo)
    
    acc = accuracy_score(y_test, preds)
    print(f"[{dataset_name}] {algo} + {embed} | Fold {fold_idx} | Acc: {acc:.4f}")
    return dataset_name, algo, embed, fold_idx, acc

def main():
    print("Loading data for Phase 2-A (7 ML - No CSL)...")
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
        futures = [executor.submit(train_and_evaluate_ml, *t) for t in tasks]
        for f in futures:
            results.append(f.result())
            
    df = pd.DataFrame(results, columns=['dataset', 'algo', 'embed', 'fold', 'acc'])
    summary = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()
    summary['config'] = summary['algo'] + '-' + summary['embed']
    
    out_dir = _ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "ml_baseline_reproduction.csv", index=False)
    print("Saved Phase 2-A!")

if __name__ == "__main__":
    main()
