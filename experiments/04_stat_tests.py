"""
experiments/04_stat_tests.py
=============================
Full statistical test battery (Table 4 in the paper).

Tests applied
-------------
Against the baseline (one-sample):
  - One-sample t-test   (H1: our mean > baseline point)
  - One-sample Wilcoxon signed-rank test
  - 95% t-confidence interval
  - Bootstrap 95% CI (10,000 resamples)
  - Cohen's d vs. baseline point
  - Binomial test on the single held-out set

Among our own models (paired):
  - McNemar's exact test (same held-out test set)
  - Paired t-test and Wilcoxon (across folds)
  - Nadeau-Bengio corrected resampled t-test
  - Friedman test + Nemenyi post-hoc (across embedding families)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
import scikit_posthocs as sp

from data.preprocess import load_fnfc, load_promise
from models.features  import TFIDFBuilder, SBERTBuilder, GloVeBuilder, HybridBuilder
from models.cost_weights import tempered_weights

OUT = Path(__file__).parents[1] / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = {"fnfc": 90.74, "promise": 79.98}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nb_corrected_t(diffs: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test (Equation 4 in the paper)."""
    n = len(diffs)
    rho = 1 / n_splits          # approx test/train ratio for k-fold
    mu = diffs.mean()
    var = diffs.var(ddof=1)
    t_stat = mu / np.sqrt((1 / n + rho) * var + 1e-15)
    p_val  = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return float(t_stat), float(p_val)


def mcnemar_pair(pA, pB, yt, label_a, label_b):
    a_ok = pA == yt; b_ok = pB == yt
    n10 = int(np.sum( a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok &  b_ok))
    n11 = int(np.sum( a_ok &  b_ok))
    n00 = int(np.sum(~a_ok & ~b_ok))
    table = [[n11, n10], [n01, n00]]
    res = mcnemar(table, exact=(n10 + n01 < 25), correction=True)
    print(f"    McNemar {label_a} vs {label_b}: "
          f"A✓B✗={n10}  A✗B✓={n01}  p={res.pvalue:.3e}  "
          f"{'*SIGNIFICANT*' if res.pvalue < 0.05 else 'n.s.'}")
    return res.pvalue


# ---------------------------------------------------------------------------
# Run all tests for one dataset
# ---------------------------------------------------------------------------

def run_tests(name: str, X: np.ndarray, y: np.ndarray,
              C: float, alpha_best: float, feature_builder_fn,
              paper_acc: float):

    print(f"\n{'='*65}")
    print(f"  Statistical tests — {name.upper()}")
    print(f"  Paper baseline best: {paper_acc}%")
    print(f"{'='*65}")

    # -----------------------------------------------------------------------
    # 1. Build 50-estimate CV distribution
    # -----------------------------------------------------------------------
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    accs_best, accs_unif = [], []
    for tr, te in rcv.split(X, y):
        fb = feature_builder_fn()
        A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
        pb = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_best)).fit(A, y[tr]).predict(B)
        pu = LinearSVC(C=C, class_weight=tempered_weights(y[tr], 0.0)).fit(A, y[tr]).predict(B)
        accs_best.append(accuracy_score(y[te], pb) * 100)
        accs_unif.append(accuracy_score(y[te], pu) * 100)
    accs_best = np.array(accs_best)
    accs_unif = np.array(accs_unif)

    # -----------------------------------------------------------------------
    # 2. One-sample tests vs. paper baseline
    # -----------------------------------------------------------------------
    print(f"\n  [ONE-SAMPLE: our distribution vs. paper point {paper_acc}%]")
    n = len(accs_best); m = accs_best.mean(); sd = accs_best.std(ddof=1); se = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(accs_best, n, replace=True)) for _ in range(10_000)]
    bci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    d = (m - paper_acc) / sd
    sh = stats.shapiro(accs_best)
    t  = stats.ttest_1samp(accs_best, paper_acc, alternative="greater")
    w  = stats.wilcoxon(accs_best - paper_acc, alternative="greater", zero_method="wilcox")

    print(f"    n={n}  mean={m:.3f}%  sd={sd:.3f}")
    print(f"    95% t-CI : [{ci[0]:.3f}, {ci[1]:.3f}]  — excludes paper? {ci[0] > paper_acc}")
    print(f"    95% boot : [{bci[0]:.3f}, {bci[1]:.3f}]")
    print(f"    Shapiro-Wilk: W={sh.statistic:.3f}  p={sh.pvalue:.3f}  "
          f"({'normal' if sh.pvalue > 0.05 else 'non-normal'})")
    print(f"    One-sample t-test:    t={t.statistic:.3f}  p={t.pvalue:.3e}")
    print(f"    One-sample Wilcoxon:  W={w.statistic:.1f}  p={w.pvalue:.3e}")
    print(f"    Cohen's d: {d:.2f}  "
          f"({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")

    # Binomial test on single held-out set
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=0, stratify=y)
    fb = feature_builder_fn()
    A_tr = fb.fit_transform(X[itr]); A_te = fb.transform(X[ite])
    y_pred_best = LinearSVC(C=C, class_weight=tempered_weights(y[itr], alpha_best)).fit(A_tr, y[itr]).predict(A_te)
    y_pred_unif = LinearSVC(C=C, class_weight=tempered_weights(y[itr], 0.0)).fit(A_tr, y[itr]).predict(A_te)
    y_pred_maj  = DummyClassifier(strategy="most_frequent").fit(X[itr], y[itr]).predict(X[ite])
    y_pred_glv  = None
    try:
        glv = GloVeBuilder()
        G_tr = glv.fit_transform(X[itr]); G_te = glv.transform(X[ite])
        y_pred_glv = LinearSVC(C=1.0, class_weight=tempered_weights(y[itr], alpha_best)).fit(G_tr, y[itr]).predict(G_te)
    except Exception:
        pass

    k = int((y_pred_best == y[ite]).sum()); N_te = len(ite)
    bt = stats.binomtest(k, N_te, paper_acc / 100, alternative="greater")
    print(f"    Binomial (held-out): {k}/{N_te}={k/N_te*100:.2f}%  p={bt.pvalue:.3e}")

    # -----------------------------------------------------------------------
    # 3. McNemar tests (paired on held-out set)
    # -----------------------------------------------------------------------
    print(f"\n  [McNEMAR TESTS on held-out set  N={N_te}]")
    mcnemar_pair(y_pred_best, y_pred_maj,  y[ite], "best", "majority")
    if y_pred_glv is not None:
        mcnemar_pair(y_pred_best, y_pred_glv, y[ite], "best", "GloVe-avg")
    mcnemar_pair(y_pred_best, y_pred_unif, y[ite], "cost-sensitive", "uniform")

    # -----------------------------------------------------------------------
    # 4. Paired CV: cost-sensitive vs. uniform  (Nadeau-Bengio corrected)
    # -----------------------------------------------------------------------
    print(f"\n  [PAIRED CV: cost-sensitive vs. uniform — 25 folds]")
    rcv25 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
    diffs = []
    for tr, te in rcv25.split(X, y):
        fb = feature_builder_fn()
        A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
        pb = LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_best)).fit(A, y[tr]).predict(B)
        pu = LinearSVC(C=C, class_weight=tempered_weights(y[tr], 0.0)).fit(A, y[tr]).predict(B)
        diffs.append(accuracy_score(y[te], pb) - accuracy_score(y[te], pu))
    diffs = np.array(diffs)
    t_pair = stats.ttest_rel(accs_best[:25], accs_unif[:25])
    w_pair = stats.wilcoxon(diffs)
    t_nb, p_nb = nb_corrected_t(diffs, n_splits=5)
    d_pair = diffs.mean() / diffs.std(ddof=1)
    print(f"    mean diff = {diffs.mean()*100:+.3f}pp")
    print(f"    Paired t-test:              p={t_pair.pvalue:.3f}")
    print(f"    Wilcoxon:                   p={w_pair.pvalue:.3f}")
    print(f"    Nadeau-Bengio corrected t:  p={p_nb:.3f}  "
          f"({'*sig*' if p_nb < 0.05 else 'n.s.'})")
    print(f"    Cohen's d (paired): {d_pair:.3f}")

    # -----------------------------------------------------------------------
    # 5. Friedman + Nemenyi across embedding families
    # -----------------------------------------------------------------------
    print(f"\n  [FRIEDMAN + NEMENYI: TF-IDF vs GloVe vs Contextual — 25 folds]")
    rcv_fr = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=11)
    acc_T, acc_G, acc_C = [], [], []
    glv_ok = True
    for tr, te in rcv_fr.split(X, y):
        # TF-IDF
        fb = feature_builder_fn(); A = fb.fit_transform(X[tr]); B = fb.transform(X[te])
        acc_T.append(accuracy_score(y[te],
            LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_best)).fit(A, y[tr]).predict(B)))
        # GloVe
        try:
            glv2 = GloVeBuilder()
            G = glv2.fit_transform(X[tr]); Gt = glv2.transform(X[te])
            acc_G.append(accuracy_score(y[te],
                LinearSVC(C=1.0, class_weight=tempered_weights(y[tr], alpha_best)).fit(G, y[tr]).predict(Gt)))
        except Exception:
            acc_G.append(np.nan); glv_ok = False
        # Contextual
        sbert = SBERTBuilder("sentence-transformers/all-MiniLM-L6-v2")
        E = sbert.fit_transform(X[tr]); Et = sbert.transform(X[te])
        acc_C.append(accuracy_score(y[te],
            LinearSVC(C=C, class_weight=tempered_weights(y[tr], alpha_best)).fit(E, y[tr]).predict(Et)))

    acc_T = np.array(acc_T); acc_G = np.array(acc_G); acc_C = np.array(acc_C)
    print(f"    TF-IDF mean:     {acc_T.mean()*100:.2f}%")
    if glv_ok: print(f"    GloVe mean:      {acc_G.mean()*100:.2f}%")
    print(f"    Contextual mean: {acc_C.mean()*100:.2f}%")

    if glv_ok and not np.any(np.isnan(acc_G)):
        fr = stats.friedmanchisquare(acc_T, acc_G, acc_C)
        print(f"    Friedman chi2={fr.statistic:.2f}  p={fr.pvalue:.3e}")
        mat = np.column_stack([acc_T, acc_G, acc_C])
        nem = sp.posthoc_nemenyi_friedman(mat)
        print(f"    Nemenyi p-values: TF-GloVe={nem.iloc[0,1]:.3f}  "
              f"TF-Ctx={nem.iloc[0,2]:.3f}  GloVe-Ctx={nem.iloc[1,2]:.3f}")
    else:
        fr = stats.friedmanchisquare(acc_T, acc_C)
        print(f"    Friedman (TF vs ctx) chi2={fr.statistic:.2f}  p={fr.pvalue:.3e}")

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    summary = {
        "dataset": name,
        "our_mean_pct": m, "our_sd": sd,
        "ci_lower": ci[0], "ci_upper": ci[1],
        "ttest_p": t.pvalue, "wilcoxon_p": w.pvalue,
        "cohen_d": d, "shapiro_p": sh.pvalue,
        "nb_corrected_p": p_nb,
        "friedman_p": fr.pvalue,
    }
    pd.DataFrame([summary]).to_csv(OUT / f"stat_summary_{name}.csv", index=False)
    print(f"\n  Summary saved → outputs/results/stat_summary_{name}.csv")


if __name__ == "__main__":
    Xf, yf = load_fnfc()
    run_tests("fnfc", Xf, yf, C=1.5, alpha_best=0.6,
              feature_builder_fn=TFIDFBuilder, paper_acc=PAPER["fnfc"])

    Xp, yp = load_promise()
    def prom_builder():
        return HybridBuilder(dense_builder=SBERTBuilder("sentence-transformers/all-mpnet-base-v2"))
    run_tests("promise", Xp, yp, C=2.0, alpha_best=0.5,
              feature_builder_fn=prom_builder, paper_acc=PAPER["promise"])
