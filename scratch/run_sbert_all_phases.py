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
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import os

from data.preprocess import load_fnfc, load_promise
from models.deep_embeddings import get_deep_embeddings, TextDataset
from models.deep_architectures import get_model
from torch.utils.data import DataLoader

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from models.custom_csl import MetaCost, AdaCost, CSKNN
from models.cost_weights import tempered_weights

DL_ALGORITHMS = ['DNN', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiCNN']
ML_BASE_ALGORITHMS = ['LinearSVM', 'SVM-RBF', 'DecisionTree', 'LogisticRegression', 'RandomForest', 'AdaBoost', 'NaiveBayes']
ML_CSL_ALGORITHMS = ['CS-SVM (Linear)', 'CS-SVM (RBF)', 'CS-DT', 'CS-LR', 'CS-RF', 'MetaCost', 'AdaCost', 'CS-KNN']
EMBEDDINGS = ['SBERT']

def train_model(model, dataset, criterion, optimizer, epochs, batch_size, device):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y_b.numpy())
    return accuracy_score(trues, preds), None, None, None

def run_dl_phase(phase_name, is_csl=False):
    print(f"\nRunning {phase_name} (SBERT ONLY)...")
    X_fnfc, y_fnfc = load_fnfc(clean=True)
    X_prom, y_prom = load_promise(clean=True)
    
    datasets = {
        'FNFC': (X_fnfc, y_fnfc, len(np.unique(y_fnfc))),
        'PROMISE': (X_prom, y_prom, len(np.unique(y_prom)))
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for ds_name, (X, y, num_classes) in datasets.items():
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for embed in EMBEDDINGS:
                print(f"[{ds_name}] Extracting {embed} for Fold {fold_idx}...")
                X_train_emb, X_test_emb, embed_dim = get_deep_embeddings(X_train, X_test, embed)
                
                train_ds = TextDataset(X_train_emb, y_train)
                test_ds = TextDataset(X_test_emb, y_test)
                
                for algo in DL_ALGORITHMS:
                    model = get_model(algo, embed_dim, num_classes)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    if is_csl:
                        # Simple inverse class frequency for CSL DL
                        classes, counts = np.unique(y_train, return_counts=True)
                        w = 1.0 / counts
                        w = w / w.sum() * len(classes)
                        weights = torch.tensor(w, dtype=torch.float32).to(device)
                        criterion = nn.CrossEntropyLoss(weight=weights)
                        
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    
                    train_model(model, train_ds, criterion, optimizer, epochs=10, batch_size=32, device=device)
                    acc, _, _, _ = evaluate_model(model, test_ds, batch_size=32, device=device)
                    print(f"[{ds_name}] {phase_name} {algo} + {embed} | Fold {fold_idx} | Acc: {acc:.4f}")
                    results.append([ds_name, algo, embed, fold_idx, acc])
    
    return pd.DataFrame(results, columns=['dataset', 'algo', 'embed', 'fold', 'acc'])

# ML Evaluators
def get_ml_base_model(algo):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier
    if algo == 'LinearSVM': return LinearSVC(random_state=42)
    elif algo == 'SVM-RBF': return SVC(kernel='rbf', random_state=42)
    elif algo == 'DecisionTree': return DecisionTreeClassifier(random_state=42)
    elif algo == 'LogisticRegression': return LogisticRegression(random_state=42, max_iter=1000)
    elif algo == 'RandomForest': return RandomForestClassifier(random_state=42)
    elif algo == 'AdaBoost': return AdaBoostClassifier(random_state=42)
    elif algo == 'NaiveBayes': return GaussianNB()

def get_ml_csl_model(algo, w_dict):
    if algo == 'CS-SVM (Linear)': return LinearSVC(class_weight=w_dict, random_state=42)
    elif algo == 'CS-SVM (RBF)': return SVC(kernel='rbf', class_weight=w_dict, random_state=42)
    elif algo == 'CS-DT': return DecisionTreeClassifier(class_weight=w_dict, random_state=42)
    elif algo == 'CS-LR': return LogisticRegression(class_weight=w_dict, random_state=42, max_iter=1000)
    elif algo == 'CS-RF': return RandomForestClassifier(class_weight=w_dict, random_state=42)
    elif algo == 'MetaCost': return MetaCost(cost_matrix=w_dict, n_estimators=10)
    elif algo == 'AdaCost': return AdaCost(cost_matrix=w_dict, n_estimators=50)
    elif algo == 'CS-KNN': return CSKNN(cost_matrix=w_dict, n_neighbors=5)

def train_eval_ml(ds_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes, is_csl=False):
    if len(X_train_emb.shape) == 3:
        X_train_emb = X_train_emb.mean(axis=1)
        X_test_emb = X_test_emb.mean(axis=1)
    
    if is_csl:
        w_dict = tempered_weights(y_train, alpha=0.5)
        model = get_ml_csl_model(algo, w_dict)
    else:
        model = get_ml_base_model(algo)
        
    model.fit(X_train_emb, y_train)
    preds = model.predict(X_test_emb)
    acc = accuracy_score(y_test, preds)
    return ds_name, algo, embed, fold_idx, acc

def run_ml_phase(phase_name, algos, is_csl=False):
    print(f"\nRunning {phase_name} (SBERT ONLY)...")
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
                for algo in algos:
                    tasks.append((ds_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes, is_csl))
                    
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(train_eval_ml, *t) for t in tasks]
        for f in futures:
            results.append(f.result())
            
    return pd.DataFrame(results, columns=['dataset', 'algo', 'embed', 'fold', 'acc'])


def append_and_save(new_df, csv_path):
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)

def main():
    # 1. Phase 1-A (DL Base)
    df_1a = run_dl_phase('Phase 1-A', is_csl=False)
    append_and_save(df_1a, 'phase_1_a/deep_learning_baseline_reproduction.csv')
    
    # 2. Phase 1-B (DL CSL)
    df_1b = run_dl_phase('Phase 1-B', is_csl=True)
    append_and_save(df_1b, 'phase_1_b/deep_learning_csl_reproduction.csv')
    
    # 3. Phase 2-A (ML Base)
    df_2a = run_ml_phase('Phase 2-A', ML_BASE_ALGORITHMS, is_csl=False)
    append_and_save(df_2a, 'phase_2_a/ml_baseline_reproduction.csv')
    
    # 4. Phase 2-B (ML CSL)
    df_2b = run_ml_phase('Phase 2-B', ML_CSL_ALGORITHMS, is_csl=True)
    append_and_save(df_2b, 'phase_2_b/csl_ml_baseline_reproduction.csv')
    
    print("\nALL SBERT RESULTS APPENDED TO CSVs SUCCESSFULLY!")

if __name__ == "__main__":
    main()
