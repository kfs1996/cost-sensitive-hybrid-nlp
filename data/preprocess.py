"""
data/preprocess.py
==================
Data loading, cleaning, and taxonomy harmonisation for FNFC and PROMISE_exp.

Datasets expected at:
  ../FNFC.csv     — columns: text, class
  ../Promise.csv  — columns: Requirement, Type

Both files use latin-1 encoding (raw FNFC has Windows line endings).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (resolved relative to this file so the package is location-independent)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
FNFC_PATH    = _ROOT / "FNFC.csv"
PROMISE_PATH = _ROOT / "Promise.csv"


# ---------------------------------------------------------------------------
# Contraction map (consistent with Kabootari et al. 2025 preprocessing)
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    r"\bwon't\b": "will not",  r"\bcan't\b": "cannot",
    r"\bdon't\b": "do not",    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",  r"\bisn't\b": "is not",
    r"\baren't\b": "are not",  r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not", r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not", r"\bhadn't\b": "had not",
    r"\bshouldn't\b": "should not", r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",   r"\bmightn't\b": "might not",
    r"\bmustn't\b": "must not",     r"\bdaren't\b": "dare not",
    r"\bshe's\b": "she is",  r"\bhe's\b": "he is",
    r"\bit's\b": "it is",    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
}


def _clean(text: str) -> str:
    """Lowercase, expand contractions, remove punctuation."""
    t = str(text).lower()
    for pat, repl in _CONTRACTIONS.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# FNFC
# ---------------------------------------------------------------------------
def load_fnfc(clean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load FNFC dataset.

    Returns
    -------
    X : ndarray of str, shape (7060,)
    y : ndarray of str, shape (7060,)  — 14 class codes
    """
    df = pd.read_csv(FNFC_PATH, encoding="latin-1").dropna().reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()
    # normalise column names (the file uses 'text' and 'class')
    text_col  = [c for c in df.columns if "text" in c or "require" in c][0]
    label_col = [c for c in df.columns if "class" in c or "type" in c or "label" in c][0]
    df["_text"] = df[text_col].astype(str).apply(_clean if clean else str)
    df["_label"] = df[label_col].astype(str).str.strip()
    return df["_text"].values, df["_label"].values


# ---------------------------------------------------------------------------
# PROMISE_exp
# ---------------------------------------------------------------------------
def load_promise(clean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PROMISE_exp dataset.

    Returns
    -------
    X : ndarray of str, shape (969,)
    y : ndarray of str, shape (969,)  — 12 class codes
    """
    df = pd.read_csv(PROMISE_PATH, encoding="latin-1") \
           .dropna(subset=["Requirement", "Type"]) \
           .reset_index(drop=True)
    df["_text"] = df["Requirement"].astype(str).apply(_clean if clean else str)
    df["_label"] = df["Type"].astype(str).str.strip()
    return df["_text"].values, df["_label"].values


# ---------------------------------------------------------------------------
# Cross-corpus: harmonise taxonomies to 11 shared classes
# ---------------------------------------------------------------------------

# Mappings that make FNFC codes equivalent to PROMISE codes
_FNFC_MAP = {"LL": "L", "M": "MN", "P": "PO"}

# Classes present in FNFC but not PROMISE (and vice-versa), plus the clash
_DROP_FNFC    = {"AU", "R", "O"}   # AU=Autonomy, R=Reliability, O=Inter-Operability
_DROP_PROMISE = {"O"}              # O=Operational (different concept; no FNFC equiv)

SHARED_CLASSES = ["A", "F", "FT", "L", "LF", "MN", "PE", "PO", "SC", "SE", "US"]


def load_harmonised(clean: bool = True) -> tuple[
    np.ndarray, np.ndarray,   # FNFC  (X_fn, y_fn)
    np.ndarray, np.ndarray,   # PROMISE (X_pr, y_pr)
]:
    """
    Load both datasets with harmonised 11-class taxonomy.

    Returns
    -------
    X_fn, y_fn  : FNFC texts and labels (6821 rows after dropping 3 classes)
    X_pr, y_pr  : PROMISE_exp texts and labels (892 rows after dropping O)
    """
    X_fn_raw, y_fn_raw = load_fnfc(clean=clean)
    mask_fn = ~pd.Series(y_fn_raw).isin(_DROP_FNFC)
    y_fn = pd.Series(y_fn_raw).replace(_FNFC_MAP).values[mask_fn]
    X_fn = X_fn_raw[mask_fn]

    X_pr_raw, y_pr_raw = load_promise(clean=clean)
    mask_pr = ~pd.Series(y_pr_raw).isin(_DROP_PROMISE)
    y_pr = y_pr_raw[mask_pr]
    X_pr = X_pr_raw[mask_pr]

    # Sanity check: only shared classes remain
    assert set(y_fn).issubset(set(SHARED_CLASSES)), \
        f"Unexpected FNFC classes after harmonisation: {set(y_fn) - set(SHARED_CLASSES)}"
    assert set(y_pr).issubset(set(SHARED_CLASSES)), \
        f"Unexpected PROMISE classes after harmonisation: {set(y_pr) - set(SHARED_CLASSES)}"

    return X_fn, y_fn, X_pr, y_pr


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Xf, yf = load_fnfc()
    Xp, yp = load_promise()
    print(f"FNFC   : {len(Xf)} rows, {len(set(yf))} classes — majority {pd.Series(yf).value_counts(normalize=True).max():.3f}")
    print(f"PROMISE: {len(Xp)} rows, {len(set(yp))} classes — majority {pd.Series(yp).value_counts(normalize=True).max():.3f}")

    Xfh, yfh, Xph, yph = load_harmonised()
    print(f"\nHarmonised FNFC   : {len(Xfh)} rows, {sorted(set(yfh))}")
    print(f"Harmonised PROMISE: {len(Xph)} rows, {sorted(set(yph))}")
