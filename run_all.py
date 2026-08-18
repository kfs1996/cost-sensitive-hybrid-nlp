"""
run_all.py
==========
Master runner — executes all seven experiment scripts in order and
produces a final summary of key metrics.

Usage
-----
    python run_all.py           # full pipeline
    python run_all.py --skip 1  # skip experiment 01 (EDA)
    python run_all.py --only 2  # run only experiment 02

Estimated runtimes on a modern CPU (no GPU):
  01_eda           ~5 s
  02_benchmark     ~15 min  (50 CV folds × 2 datasets, sentence encoding)
  03_cost_sweep    ~10 min
  04_stat_tests    ~20 min  (GloVe download ~6 min first run only)
  05_embed_compare ~15 min
  06_lift_all24    ~1 s     (uses hard-coded paper numbers only)
  07_cross_corpus  ~8 min
  Total            ~70 min  (first run, including model downloads)
                   ~50 min  (subsequent runs, models cached)
"""

import argparse
import importlib
import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    _orig_send = requests.Session.send
    def _patched_send(self, request, **kwargs):
        kwargs['verify'] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _patched_send
except Exception:
    pass

try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
from pathlib import Path

EXPERIMENTS = [
    ("01", "01_eda",           "EDA dashboards"),
    ("02", "02_benchmark",     "Main benchmark (Tables 2–3)"),
    ("03", "03_cost_sweep",    "Cost-sensitivity sweep (Fig. 4)"),
    ("04", "04_stat_tests",    "Statistical significance battery (Table 4)"),
    ("05", "05_embed_compare", "Embedding-by-embedding comparison (Table 3)"),
    ("06", "06_lift_all24",    "Lift over all 24 configurations (Figs.)"),
    ("07", "07_cross_corpus",  "Cross-corpus generalisation (Table 5)"),
    ("08", "08_ablation",      "Ablation study and hyperparameter tuning"),
]

# Target key numbers to verify at the end
EXPECTED = {
    "FNFC   best acc (5-fold CV)":     ("≥90.74%",  "our best should exceed paper's 90.74%"),
    "PROMISE best acc (5-fold CV)":    ("≥79.98%",  "our best should exceed paper's 79.98%"),
    "Cost-sensitive alpha sweet spot": ("0.5–0.8",  "Pareto improvement in this range"),
    "Friedman p-value (embeddings)":   ("<0.001",   "embedding choice is statistically significant"),
    "Cross-corpus best direction":     ("ctx+cost", "contextual + cost-sens for FNFC→PROMISE"),
}


def run_experiment(module_stem: str, description: str) -> bool:
    """Dynamically import and run an experiment module's __main__ block."""
    print(f"\n{'='*65}")
    print(f"  {description}")
    print(f"{'='*65}")
    t0 = time.time()
    try:
        exp_path = Path(__file__).parent / "experiments"
        sys.path.insert(0, str(exp_path.parent))
        sys.path.insert(0, str(exp_path))

        # Run as subprocess so each script's __main__ guard fires correctly
        import subprocess
        result = subprocess.run(
            [sys.executable, str(exp_path / f"{module_stem}.py")],
            capture_output=False,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  ✓ Completed in {elapsed:.0f}s")
            return True
        else:
            print(f"\n  ✗ FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
            return False
    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--skip", nargs="*", metavar="N",
                        help="Experiment numbers to skip (e.g. --skip 1 4)")
    parser.add_argument("--only", nargs="*", metavar="N",
                        help="Only run these experiment numbers (e.g. --only 2 3)")
    args = parser.parse_args()

    skip = set(args.skip or [])
    only = set(args.only or [])

    # Ensure output directories exist
    (Path(__file__).parent / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "outputs" / "results").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  Software Requirements Classification — Full Pipeline")
    print("="*65)
    print("  Outputs will be written to:")
    print("    outputs/figures/   — all PNG figures")
    print("    outputs/results/   — CSV tables and per-class reports")

    total_start = time.time()
    statuses = {}

    for num, stem, desc in EXPERIMENTS:
        if only and num not in only:
            continue
        if num in skip:
            print(f"\n  [skipping {num} — {desc}]")
            statuses[num] = "skipped"
            continue
        ok = run_experiment(stem, desc)
        statuses[num] = "ok" if ok else "FAILED"

    elapsed_total = time.time() - total_start

    # Summary
    print("\n" + "="*65)
    print("  PIPELINE SUMMARY")
    print("="*65)
    for num, stem, desc in EXPERIMENTS:
        status = statuses.get(num, "—")
        icon = "✓" if status == "ok" else ("—" if status == "skipped" else "✗")
        print(f"  {icon}  {num}  {desc}")
    print(f"\n  Total elapsed: {elapsed_total/60:.1f} min")
    print("\n  Key results to verify:")
    for k, (val, note) in EXPECTED.items():
        print(f"    {k}: {val}  ({note})")
    print("\n  Outputs:")
    fig_dir = Path(__file__).parent / "outputs" / "figures"
    res_dir = Path(__file__).parent / "outputs" / "results"
    for p in sorted(fig_dir.glob("*.png")):
        print(f"    {p.relative_to(Path(__file__).parent)}")
    for p in sorted(res_dir.glob("*")):
        print(f"    {p.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
