import pandas as pd
df = pd.read_csv('phase_2_b/csl_ml_baseline_reproduction.csv')
fnfc = df[df['dataset'] == 'FNFC']
best = fnfc.groupby('algo')['acc'].max().sort_values(ascending=False)
print("ALGORITHM-WISE PEAK FOR FNFC (PHASE 2-B):")
print(best)
