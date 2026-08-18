import pandas as pd

df = pd.read_csv('phase_2_b/csl_ml_baseline_reproduction.csv')
print("\n--- Phase 2-B Absolute Max Accuracies ---")
for dataset in ['FNFC', 'PROMISE']:
    sub = df[df['dataset'] == dataset]
    best = sub.loc[sub['acc'].idxmax()]
    print(f"{dataset}: {best['acc']:.4f} ({best['algo']} + {best['embed']})")
