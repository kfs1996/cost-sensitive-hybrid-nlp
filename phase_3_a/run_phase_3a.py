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
import os
import itertools

from data.preprocess import load_fnfc, load_promise
from models.deep_embeddings import get_hybrid_embeddings, TextDataset
from models.deep_architectures import get_model
from torch.utils.data import DataLoader

DL_ALGORITHMS = ['DNN', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiCNN']
BASE_EMBEDDINGS = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'SBERT']

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
    return accuracy_score(trues, preds)

def main():
    print("Initializing Phase 3-A: HYBRID DEEP EMBEDDINGS (All 26 Combinations)")
    X_fnfc, y_fnfc = load_fnfc(clean=True)
    X_prom, y_prom = load_promise(clean=True)
    
    datasets = {
        'FNFC': (X_fnfc, y_fnfc, len(np.unique(y_fnfc))),
        'PROMISE': (X_prom, y_prom, len(np.unique(y_prom)))
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Loaded {len(datasets)} datasets. Generating 10 Hybrid Pair Combinations...")
    
    # User requested exactly the 10 pairs
    combinations = list(itertools.combinations(BASE_EMBEDDINGS, 2))
    
    results = []
    
    out_dir = _ROOT / "phase_3_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hybrid_dl_baseline_reproduction.csv"
    
    # Initialize empty CSV with headers
    pd.DataFrame(columns=['dataset', 'algo', 'embed', 'fold', 'acc']).to_csv(csv_path, index=False)
    
    for ds_name, (X, y, num_classes) in datasets.items():
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for combo in combinations:
                combo_name = "+".join(combo)
                print(f"[{ds_name}] Extracting {combo_name} for Fold {fold_idx}...")
                
                try:
                    X_train_emb, X_test_emb, embed_dim = get_hybrid_embeddings(X_train, X_test, combo)
                except Exception as e:
                    print(f"Failed extracting {combo_name}: {e}")
                    continue
                
                train_ds = TextDataset(X_train_emb, y_train)
                test_ds = TextDataset(X_test_emb, y_test)
                
                for algo in DL_ALGORITHMS:
                    model = get_model(algo, embed_dim, num_classes)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    
                    try:
                        train_model(model, train_ds, criterion, optimizer, epochs=10, batch_size=32, device=device)
                        acc = evaluate_model(model, test_ds, batch_size=32, device=device)
                        print(f"[{ds_name}] {algo} + {combo_name} | Fold {fold_idx} | Acc: {acc:.4f}")
                        
                        # Save Progressively!
                        new_row = pd.DataFrame([[ds_name, algo, combo_name, fold_idx, acc]], 
                                               columns=['dataset', 'algo', 'embed', 'fold', 'acc'])
                        new_row.to_csv(csv_path, mode='a', header=False, index=False)
                        
                    except Exception as e:
                        print(f"Failed training {algo} with {combo_name}: {e}")
                        
    print("Saved Phase 3-A (Hybrid Deep Learning)!")

if __name__ == "__main__":
    main()
