import pandas as pd
import numpy as np

def get_peak(file_path, dataset):
    try:
        df = pd.read_csv(file_path)
        df_ds = df[df['dataset'] == dataset]
        if df_ds.empty: return 0.0
        
        # calculate 5-fold mean
        # in phase 2-a and 2-b, grouping might just be algo and embed
        df_avg = df_ds.groupby(['algo', 'embed'])['acc'].mean().reset_index()
        return df_avg['acc'].max() * 100
    except Exception as e:
        return 0.0

datasets = ['FNFC', 'PROMISE']
phases = {
    'Phase 1-A (Deep Learning Baseline)': 'phase_1_a/deep_learning_baseline_reproduction.csv',
    'Phase 1-B (Deep Learning CSL)': 'phase_1_b/deep_learning_csl_reproduction.csv',
    'Phase 2-A (Native ML)': 'phase_2_a/ml_baseline_reproduction.csv',
    'Phase 2-B (Cost-Sensitive ML)': 'phase_2_b/csl_ml_baseline_reproduction.csv',
    'Phase 3-B (2-Way Hybrid CSL)': 'phase_3_b/hybrid_csl_ml_reproduction.csv',
    'Phase 3-C (3-Way Hybrid CSL)': 'phase_3_c/tri_hybrid_csl_ml_reproduction.csv',
}

base_paper = {'FNFC': 90.74, 'PROMISE': 79.98}

print("=== DYNAMIC PHASE-BY-PHASE COMPARISON TO BASE PAPER ===")
for ds in datasets:
    print(f"\n[{ds} Dataset] (Base Paper Peak: {base_paper[ds]}%)")
    for phase_name, file_path in phases.items():
        peak = get_peak(file_path, ds)
        if peak == 0.0:
            print(f"  {phase_name}: Failed / Incomplete")
            continue
            
        diff = peak - base_paper[ds]
        status = f"BEAT BASE PAPER BY +{diff:.2f}%" if diff > 0 else f"Below base paper ({diff:.2f}%)"
        print(f"  {phase_name}: {peak:.2f}% -> {status}")

print(f"  Phase 3-A (Hybrid Deep Learning): 0.00% -> Failed (Out of Memory OOM)")
