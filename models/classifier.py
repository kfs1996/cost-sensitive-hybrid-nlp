"""
models/classifier.py
====================
Cost-sensitive LinearSVC wrapper with cross-validation helpers.

The classifier implements Equation 2 (one-vs-rest SVM) and Equation 3
(tempered class weights) from the paper.
"""

from __future__ import annotations
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.cost_weights import tempered_weights


# ===========================================================================
# Core classifier
# ===========================================================================

class CostSensitiveSVM:
    """
    One-vs-rest LinearSVC with tempered inverse-frequency class weights.

    Parameters
    ----------
    C     : regularisation parameter (default 1.5 for FNFC, 2.0 for PROMISE)
    alpha : cost-sensitivity exponent (default 0.5; 0=uniform, 1=balanced)
    """

    def __init__(self, C: float = 1.5, alpha: float = 0.5):
        self.C = C
        self.alpha = alpha
        self._clf = None

    def fit(self, X, y: np.ndarray) -> "CostSensitiveSVM":
        w = tempered_weights(y, self.alpha)
        self._clf = LinearSVC(C=self.C, class_weight=w)
        self._clf.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit first.")
        return self._clf.predict(X)
        
    def decision_function(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit first.")
        return self._clf.decision_function(X)

    def score(self, X, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

# ===========================================================================
# Cost‑sensitive Decision Tree
# ===========================================================================

from sklearn.tree import DecisionTreeClassifier

class CostSensitiveDecisionTree:
    """DecisionTreeClassifier with tempered class weights.

    Parameters
    ----------
    max_depth : int | None, default=None
        Depth of the tree; ``None`` means expand until all leaves are pure.
    alpha : float, default=0.5
        Cost‑sensitivity exponent (tempered weights).
    """

    def __init__(self, max_depth: int | None = None, alpha: float = 0.5):
        self.max_depth = max_depth
        self.alpha = alpha
        self._clf = None

    def fit(self, X, y: np.ndarray) -> "CostSensitiveDecisionTree":
        w = tempered_weights(y, self.alpha)
        self._clf = DecisionTreeClassifier(max_depth=self.max_depth, class_weight=w)
        self._clf.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit first.")
        return self._clf.predict(X)

    def score(self, X, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

# ===========================================================================
# Cost‑sensitive Logistic Regression
# ===========================================================================

from sklearn.linear_model import LogisticRegression

class CostSensitiveLogisticRegression:
    """LogisticRegression with tempered class weights.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    alpha : float, default=0.5
        Cost‑sensitivity exponent.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    """

    def __init__(self, C: float = 1.0, alpha: float = 0.5, max_iter: int = 1000):
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter
        self._clf = None

    def fit(self, X, y: np.ndarray) -> "CostSensitiveLogisticRegression":
        w = tempered_weights(y, self.alpha)
        self._clf = LogisticRegression(C=self.C, class_weight=w, max_iter=self.max_iter, solver="lbfgs")
        self._clf.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit first.")
        return self._clf.predict(X)

    def score(self, X, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


# ===========================================================================
# Cross-validation utilities
# ===========================================================================

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy":    accuracy_score(y_true, y_pred) * 100,
        "macro_f1":    f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
    }


def cross_validate(
    feature_builder,
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.5,
    alpha: float = 0.5,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """
    Repeated stratified k-fold cross-validation.

    Parameters
    ----------
    feature_builder : object with fit_transform(X_train) and transform(X_test)
    X               : raw text array
    y               : label array
    C, alpha        : SVM hyperparameters
    n_splits        : number of CV folds
    n_repeats       : number of CV repetitions (set >1 for significance tests)
    random_state    : seed for reproducibility

    Returns
    -------
    dict with keys 'accuracy', 'macro_f1', 'weighted_f1' — each a 1D array
    of length n_splits * n_repeats.
    """
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)

    results = {"accuracy": [], "macro_f1": [], "weighted_f1": []}

    for tr, te in cv.split(X, y):
        fb = feature_builder()          # fresh builder per fold
        A_tr = fb.fit_transform(X[tr])
        A_te = fb.transform(X[te])

        clf = CostSensitiveSVM(C=C, alpha=alpha)
        clf.fit(A_tr, y[tr])
        pred = clf.predict(A_te)

        m = _metrics(y[te], pred)
        for k in results:
            results[k].append(m[k])

    return {k: np.array(v) for k, v in results.items()}


def holdout_predict(
    feature_builder,
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.5,
    alpha: float = 0.5,
    test_size: float = 0.20,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single 80/20 stratified train/test split (paper's protocol).

    Returns
    -------
    y_test : true labels on the test partition
    y_pred : predicted labels
    test_idx : indices of test rows in the original array
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=test_size,
                                random_state=random_state, stratify=y)
    fb = feature_builder()
    A_tr = fb.fit_transform(X[itr])
    A_te = fb.transform(X[ite])

    clf = CostSensitiveSVM(C=C, alpha=alpha)
    clf.fit(A_tr, y[itr])
    y_pred = clf.predict(A_te)

    return y[ite], y_pred, ite


def print_cv_summary(
    results: dict[str, np.ndarray],
    label: str = "Model",
    baseline_acc: float | None = None,
) -> None:
    acc = results["accuracy"]
    mf1 = results["macro_f1"]
    wf1 = results["weighted_f1"]
    n = len(acc)
    print(f"\n{label}  [{n} CV estimates]")
    print(f"  Accuracy   : {acc.mean():.2f}% ± {acc.std():.2f}")
    print(f"  Macro-F1   : {mf1.mean():.2f}% ± {mf1.std():.2f}")
    print(f"  Weighted-F1: {wf1.mean():.2f}% ± {wf1.std():.2f}")
    if baseline_acc is not None:
        delta = acc.mean() - baseline_acc
        print(f"  vs baseline {baseline_acc:.2f}%:  {delta:+.2f} pts")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from data.preprocess import load_fnfc
    from models.features import TFIDFBuilder

    X, y = load_fnfc()
    print("Running quick 5-fold CV on FNFC with TF-IDF + CostSensitiveSVM …")
    res = cross_validate(TFIDFBuilder, X, y, C=1.5, alpha=0.5)
    print_cv_summary(res, label="TF-IDF + SVM (alpha=0.5)", baseline_acc=90.74)
