"""
models/ensemble.py
==================
Cost-Aware Stacking Meta-Classifier implementation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from models.classifier import CostSensitiveSVM
from models.cost_weights import tempered_weights

class CostAwareStackingClassifier:
    """
    Custom Heterogeneous Stacking ensemble with Meta-Level Cost-Weighting.
    """
    def __init__(self, base_models, meta_classifier="logreg", alpha=0.5, cv=5, stacking_mode="soft", tune_meta=False):
        """
        base_models: list of tuples (name, feature_builder_class, clf_type)
        meta_classifier: 'logreg', 'rf', or 'svm'
        alpha: the cost-weighting exponent
        cv: internal folds for out-of-fold predictions
        stacking_mode: 'hard' (discrete labels) or 'soft' (continuous probabilities/decision functions)
        tune_meta: whether to apply GridSearchCV to the meta-classifier
        """
        self.base_models = base_models
        self.meta_classifier_type = meta_classifier
        self.alpha = alpha
        self.cv = cv
        self.stacking_mode = stacking_mode
        self.tune_meta = tune_meta
        
        self.trained_base_features = []
        self.trained_base_clfs = []
        self.meta_clf = None
        self.classes_ = None

    def _get_base_clf(self, type_str):
        if type_str == "svm":
            return CostSensitiveSVM(C=1.5, alpha=self.alpha)
        # Fallback to SVM for all for now, to keep base models strong
        return CostSensitiveSVM(C=1.5, alpha=self.alpha)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(X)
        n_base = len(self.base_models)
        if self.stacking_mode == "soft":
            n_features_per_base = 1 if len(self.classes_) <= 2 else len(self.classes_)
            meta_features = np.zeros((n_samples, n_base * n_features_per_base), dtype=float)
        else:
            meta_features = np.zeros((n_samples, n_base), dtype=int)
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # 1. Generate out-of-fold continuous predictions for the meta-classifier training
        for tr, te in skf.split(X, y):
            for m_idx, (name, builder_class, clf_type) in enumerate(self.base_models):
                fb = builder_class()
                X_tr_feat = fb.fit_transform(X[tr])
                X_te_feat = fb.transform(X[te])
                
                clf = self._get_base_clf(clf_type)
                clf.fit(X_tr_feat, y[tr])
                
                if self.stacking_mode == "soft":
                    d_scores = clf.decision_function(X_te_feat)
                    if d_scores.ndim == 1:
                        d_scores = d_scores.reshape(-1, 1)
                    start_col = m_idx * n_features_per_base
                    end_col = start_col + n_features_per_base
                    meta_features[te, start_col:end_col] = d_scores
                else:
                    preds = clf.predict(X_te_feat)
                    meta_features[te, m_idx] = np.searchsorted(self.classes_, preds)
                
        # 2. Train the final base models on ALL training data
        for m_idx, (name, builder_class, clf_type) in enumerate(self.base_models):
            fb = builder_class()
            X_feat = fb.fit_transform(X)
            clf = self._get_base_clf(clf_type)
            clf.fit(X_feat, y)
            
            self.trained_base_features.append(fb)
            self.trained_base_clfs.append(clf)
            
        # 3. Train the Meta-Classifier on the out-of-fold predictions
        # Calculate cost-weights if alpha > 0
        w = tempered_weights(y, self.alpha) if self.alpha > 0.0 else None
        
        if self.meta_classifier_type == "logreg":
            base_meta = LogisticRegression(class_weight=w, max_iter=1000, random_state=42)
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
        elif self.meta_classifier_type == "rf":
            base_meta = RandomForestClassifier(class_weight=w, random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}
        elif self.meta_classifier_type == "svm":
            base_meta = LinearSVC(class_weight=w, random_state=42)
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
        else:
            raise ValueError(f"Unknown meta-classifier: {self.meta_classifier_type}")
            
        if self.tune_meta:
            from sklearn.model_selection import GridSearchCV
            self.meta_clf = GridSearchCV(base_meta, param_grid, cv=3, n_jobs=1, scoring='accuracy')
        else:
            self.meta_clf = base_meta
            
        self.meta_clf.fit(meta_features, y)
        return self

    def predict(self, X):
        n_samples = len(X)
        n_base = len(self.base_models)
        
        if self.stacking_mode == "soft":
            n_features_per_base = 1 if len(self.classes_) <= 2 else len(self.classes_)
            meta_features = np.zeros((n_samples, n_base * n_features_per_base), dtype=float)
        else:
            meta_features = np.zeros((n_samples, n_base), dtype=int)
        
        for m_idx in range(n_base):
            fb = self.trained_base_features[m_idx]
            clf = self.trained_base_clfs[m_idx]
            X_feat = fb.transform(X)
            
            if self.stacking_mode == "soft":
                d_scores = clf.decision_function(X_feat)
                if d_scores.ndim == 1:
                    d_scores = d_scores.reshape(-1, 1)
                start_col = m_idx * n_features_per_base
                end_col = start_col + n_features_per_base
                meta_features[:, start_col:end_col] = d_scores
            else:
                preds = clf.predict(X_feat)
                meta_features[:, m_idx] = np.searchsorted(self.classes_, preds)
            
        return self.meta_clf.predict(meta_features)

def cross_validate_ensemble(
    X, y,
    base_models,
    meta_classifier="logreg",
    alpha=0.5,
    stacking_mode="soft",
    n_splits=5,
    tune_meta=False,
    random_state=42
):
    """Evaluates the stacking model using StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metrics = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/{n_splits}...")
        stack = CostAwareStackingClassifier(
            base_models=base_models,
            meta_classifier=meta_classifier,
            alpha=alpha,
            cv=3, # internal cv for stacking
            stacking_mode=stacking_mode,
            tune_meta=tune_meta
        )
        stack.fit(X[tr], y[tr])
        preds = stack.predict(X[te])
        
        metrics["accuracy"].append(accuracy_score(y[te], preds) * 100)
        metrics["macro_f1"].append(f1_score(y[te], preds, average="macro", zero_division=0) * 100)
        metrics["weighted_f1"].append(f1_score(y[te], preds, average="weighted", zero_division=0) * 100)
        
    return {k: np.array(v) for k, v in metrics.items()}
