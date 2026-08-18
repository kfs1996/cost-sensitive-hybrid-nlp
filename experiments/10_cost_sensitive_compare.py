# experiments/10_cost_sensitive_compare.py
"""
Compare cost‑sensitive classifiers (Linear SVM, Decision Tree, Logistic Regression)
using the Phase 1 single‑model pipelines on the FNFC and PROMISE datasets.
Results are saved to `outputs/results/cost_sensitive_comparison.csv`.
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from pathlib import Path

# Windows settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from data.preprocess import load_fnfc, load_promise
from models.features import TFIDFBuilder, SBERTBuilder, HybridBuilder
from models.classifier import (
    CostSensitiveSVM,
    CostSensitiveDecisionTree,
    CostSensitiveLogisticRegression,
)
from models.classifier import cross_validate as svm_cv

# Helper to run CV with arbitrary classifier
def cv_generic(feature_builder, classifier_cls, X, y, C, alpha, n_splits=5, random_state=42):
    """Cross‑validate a generic cost‑sensitive classifier.
    classifier_cls must implement `fit(X, y)` and `predict(X)`.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    for tr, te in cv.split(X, y):
        fb = feature_builder()  # fresh builder per fold
        A_tr = fb.fit_transform(X[tr])
        A_te = fb.transform(X[te])
        # instantiate classifier
        if classifier_cls is CostSensitiveSVM:
            clf = classifier_cls(C=C, alpha=alpha)
        elif classifier_cls is CostSensitiveDecisionTree:
            clf = classifier_cls(alpha=alpha)
        elif classifier_cls is CostSensitiveLogisticRegression:
            clf = classifier_cls(C=C, alpha=alpha)
        else:
            raise ValueError("Unsupported classifier")
        clf.fit(A_tr, y[tr])
        pred = clf.predict(A_te)
        acc = accuracy_score(y[te], pred) * 100
        macro_f1 = f1_score(y[te], pred, average="macro", zero_division=0) * 100
        weighted_f1 = f1_score(y[te], pred, average="weighted", zero_division=0) * 100
        results["accuracy"].append(acc)
        results["macro_f1"].append(macro_f1)
        results["weighted_f1"].append(weighted_f1)
    return {k: np.array(v) for k, v in results.items()}

# Configuration
DATASETS = {"FNFC": load_fnfc, "PROMISE": load_promise}
SINGLE_MODELS = [
    ("TFIDF", TFIDFBuilder),
    ("BERT", SBERTBuilder),
    ("Hybrid", lambda: HybridBuilder(dense_builder=SBERTBuilder())),
]
CLASSIFIERS = [
    ("LinearSVM", CostSensitiveSVM),
    ("DecisionTree", CostSensitiveDecisionTree),
    ("LogisticReg", CostSensitiveLogisticRegression),
]
ALPHA = 0.5

out_dir = Path(__file__).parents[2] / "outputs" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "cost_sensitive_comparison.csv"

results = []
for ds_name, load_fn in DATASETS.items():
    print(f"\n=== Dataset: {ds_name} ===")
    X, y = load_fn()
    C = 2.0 if ds_name == "PROMISE" else 1.5
    for model_name, builder in SINGLE_MODELS:
        for clf_name, clf_cls in CLASSIFIERS:
            print(f"Running {model_name} + {clf_name} (alpha={ALPHA})")
            start = time.time()
            if clf_cls is CostSensitiveSVM:
                # reuse existing helper for speed
                res = svm_cv(builder, X, y, C=C, alpha=ALPHA, n_splits=5)
            else:
                res = cv_generic(builder, clf_cls, X, y, C=C, alpha=ALPHA, n_splits=5)
            end = time.time()
            results.append({
                "Dataset": ds_name,
                "Feature": model_name,
                "Classifier": clf_name,
                "Alpha": ALPHA,
                "Accuracy": np.mean(res["accuracy"]),
                "Macro_F1": np.mean(res["macro_f1"]),
                "Weighted_F1": np.mean(res["weighted_f1"]),
                "Time_sec": end - start,
            })
            pd.DataFrame(results).to_csv(out_file, index=False)

print(f"Results written to {out_file}")
