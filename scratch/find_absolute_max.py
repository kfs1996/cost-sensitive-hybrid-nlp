import pandas as pd
max_acc_fnfc = 0.0
best_info_fnfc = ''
max_acc_prom = 0.0
best_info_prom = ''

files = [
    'phase_1_a/deep_learning_baseline_reproduction.csv', 
    'phase_1_b/deep_learning_csl_reproduction.csv', 
    'phase_2_a/ml_baseline_reproduction.csv', 
    'phase_2_b/csl_ml_baseline_reproduction.csv'
]

for f in files:
    try:
        df = pd.read_csv(f)
        for idx, row in df.iterrows():
            if row['dataset'] == 'FNFC':
                if row['acc'] > max_acc_fnfc:
                    max_acc_fnfc = row['acc']
                    best_info_fnfc = f"{row['algo']} + {row['embed']} in {f}"
            elif row['dataset'] == 'PROMISE':
                if row['acc'] > max_acc_prom:
                    max_acc_prom = row['acc']
                    best_info_prom = f"{row['algo']} + {row['embed']} in {f}"
    except:
        pass
print(f'FNFC ABSOLUTE MAX: {max_acc_fnfc:.4f} -> {best_info_fnfc}')
print(f'PROMISE ABSOLUTE MAX: {max_acc_prom:.4f} -> {best_info_prom}')
