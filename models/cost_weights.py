"""
models/cost_weights.py
======================
Tempered inverse-frequency class weighting (Equation 3 in the paper).

    w_c = (N / (K * n_c)) ** alpha

where
  N     = total training instances
  K     = number of classes
  n_c   = training instances in class c
  alpha = exponent in [0, 1]
              0 → uniform weights (no cost-sensitivity)
              1 → sklearn 'balanced' (full inverse-frequency)
     0 < alpha < 1 → tempered interpolation (recommended: 0.5–0.6)
"""

from __future__ import annotations
from collections import Counter
import numpy as np


def tempered_weights(y_train: np.ndarray, alpha: float = 0.5) -> dict[str, float]:
    """
    Compute per-class cost weights for the training labels.

    Parameters
    ----------
    y_train : array-like of class labels (training fold only)
    alpha   : float in [0, 1]

    Returns
    -------
    dict mapping class label -> weight (float)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    counts = Counter(y_train)
    K = len(counts)
    N = len(y_train)
    return {c: (N / (K * n)) ** alpha for c, n in counts.items()}


def sweep_alpha(
    y_train: np.ndarray,
    alphas: list[float] | None = None,
) -> dict[float, dict[str, float]]:
    """
    Compute weights for multiple alpha values at once.

    Returns
    -------
    dict mapping alpha -> {class -> weight}
    """
    if alphas is None:
        alphas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    return {a: tempered_weights(y_train, a) for a in alphas}


def print_weight_table(y_train: np.ndarray, alpha: float = 0.5) -> None:
    """Pretty-print the class weights and instance counts."""
    w = tempered_weights(y_train, alpha)
    counts = Counter(y_train)
    print(f"\nClass weights (alpha={alpha}):")
    print(f"  {'class':6s}  {'n':>6s}  {'weight':>8s}")
    for c in sorted(w, key=lambda x: -counts[x]):
        print(f"  {c:6s}  {counts[c]:>6d}  {w[c]:>8.4f}")


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    from data.preprocess import load_fnfc
    _, y = load_fnfc()
    print_weight_table(y, alpha=0.6)
