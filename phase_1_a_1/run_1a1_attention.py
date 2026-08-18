import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add package root to sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor

from data.preprocess import load_fnfc, load_promise
from models.deep_architectures import get_model
from models.deep_embeddings import TextDataset, get_deep_embeddings

ALGORITHMS = ['DNN', 'CNN', 'BiCNN', 'LSTM', 'BiLSTM', 'GRU']
EMBEDDINGS = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'MPNet']

def train_and_evaluate(dataset_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes, embed_dim):
    """
    Train a single deep learning model with Early Stopping.
    """
    # CRITICAL SPEEDUP: Prevent PyTorch from spawning internal threads that fight with ProcessPoolExecutor
    torch.set_num_threads(1)
    
    # 2. Datasets
    train_ds = TextDataset(X_train_emb, y_train)
    test_ds = TextDataset(X_test_emb, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # 3. Model
    model = get_model(algo, embed_dim, num_classes, use_attention=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Early Stopping Training Loop
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                val_loss += criterion(out, batch_y).item()
                
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            # print(f"Early stopping at epoch {epoch}")
            break
            
    # 5. Final Evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out = model(batch_x)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.numpy())
            trues.extend(batch_y.numpy())
            
    acc = accuracy_score(trues, preds)
    print(f"[{dataset_name}] {algo} + {embed} | Fold {fold_idx} | Acc: {acc:.4f}")
    return dataset_name, algo, embed, fold_idx, acc

def main():
    print("Loading data...")
    X_fnfc, y_fnfc = load_fnfc(clean=True)
    X_prom, y_prom = load_promise(clean=True)
    
    datasets = {
        'FNFC': (X_fnfc, y_fnfc, len(np.unique(y_fnfc))),
        'PROMISE': (X_prom, y_prom, len(np.unique(y_prom)))
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tasks = []
    
    for ds_name, (X, y, num_classes) in datasets.items():
        # Full dataset run

        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for embed in EMBEDDINGS:
                print(f"[{ds_name} - Fold {fold_idx}] Extracting {embed} representations...")
                X_train_emb, X_test_emb, embed_dim = get_deep_embeddings(X_train, X_test, embed)
                
                for algo in ALGORITHMS:
                    tasks.append((ds_name, algo, embed, fold_idx, X_train_emb, y_train, X_test_emb, y_test, num_classes, embed_dim))
    
    print(f"Submitting {len(tasks)} model training jobs to parallel pool (max threads)...")
    
    results = []
    # Use max_workers=2 to prevent memory explosions on Deep Learning
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_and_evaluate, *t)
            for t in tasks
        ]
        for f in futures:
            results.append(f.result())
            
    # Aggregate results
    df = pd.DataFrame(results, columns=['dataset', 'algo', 'embed', 'fold', 'acc'])
    summary = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()
    summary['config'] = summary['algo'] + '-' + summary['embed']
    
    out_dir = _ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "attention_dl_reproduction.csv", index=False)
    print("Done! Saved to attention_dl_reproduction.csv")

if __name__ == "__main__":
    main()
