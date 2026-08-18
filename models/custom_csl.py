import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

class MetaCost(BaseEstimator, ClassifierMixin):
    """
    Pedro Domingos' MetaCost Algorithm (1999).
    Relabels the training data using a Bagging classifier to minimize expected cost.
    """
    def __init__(self, base_estimator=None, cost_matrix=None, n_estimators=10):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(random_state=42)
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        bag = BaggingClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators, random_state=42)
        bag.fit(X, y)
        probs = bag.predict_proba(X)
        classes = bag.classes_
        
        y_relabeled = np.zeros(len(y), dtype=y.dtype)
        for i in range(len(y)):
            expected_costs = []
            for j, c in enumerate(classes):
                expected_costs.append(probs[i, j] * self.cost_matrix.get(c, 1.0))
            y_relabeled[i] = classes[np.argmax(expected_costs)]
            
        self.final_estimator_ = clone(self.base_estimator)
        self.final_estimator_.fit(X, y_relabeled)
        self.classes_ = self.final_estimator_.classes_
        return self

    def predict(self, X):
        return self.final_estimator_.predict(X)


class AdaCost(BaseEstimator, ClassifierMixin):
    """
    Simulated AdaCost wrapper. Uses AdaBoost but injects the Cost Matrix 
    as the initial sample weight distribution.
    """
    def __init__(self, cost_matrix=None, n_estimators=50):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        self.model_ = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=42)
        sample_weight = np.array([self.cost_matrix.get(label, 1.0) for label in y])
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


class CSKNN(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive k-Nearest Neighbors.
    Scales the distance-based votes by the cost matrix of the neighbor's class.
    """
    def __init__(self, cost_matrix=None, n_neighbors=5):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.y_train_ = np.array(y)
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_.fit(X, y)
        return self
        
    def predict(self, X):
        distances, indices = self.knn_.kneighbors(X)
        y_pred = []
        for i in range(len(X)):
            votes = {c: 0.0 for c in self.classes_}
            for j in range(self.n_neighbors):
                neighbor_class = self.y_train_[indices[i, j]]
                dist = distances[i, j]
                weight = 1.0 / (dist + 1e-5)
                votes[neighbor_class] += weight * self.cost_matrix.get(neighbor_class, 1.0)
            
            y_pred.append(max(votes, key=votes.get))
        return np.array(y_pred)
