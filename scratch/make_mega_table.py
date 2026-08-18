import pandas as pd
import numpy as np
import re
from pathlib import Path

# Parse base paper accuracies from comprehensive_comparison.md
base_paper_data = []
current_dataset = None

try:
    with open(r'C:\Users\Fskir\.gemini\antigravity\brain\36ba3f84-34f2-4102-99ae-df58671f3ac7\comprehensive_comparison.md', 'r', encoding='utf-8') as f:
        for line in f:
            if 'FNFC Dataset' in line:
                current_dataset = 'FNFC'
            elif 'PROMISE_exp Dataset' in line:
                current_dataset = 'PROMISE'
            elif line.startswith('| **'):
                match = re.search(r'\|\s*\*\*(.*?)\s*\+\s*(.*?)\*\*\s*\|\s*(.*?)\s*\|', line)
                if match:
                    algo = match.group(1).strip()
                    embed = match.group(2).strip()
                    base_val_raw = match.group(3)
                    
                    pct_match = re.search(r'([\d\.]+)%', base_val_raw)
                    if pct_match:
                        acc = float(pct_match.group(1)) / 100.0
                        base_paper_data.append({
                            'dataset': current_dataset,
                            'algo': algo,
                            'embed': embed,
                            'Base Paper': acc
                        })
except Exception as e:
    print(f"Error parsing base paper: {e}")

df_base = pd.DataFrame(base_paper_data)
if not df_base.empty:
    df_base['config'] = df_base['algo'] + ' + ' + df_base['embed']

df_1a = pd.read_csv('phase_1_a/deep_learning_baseline_reproduction.csv')
df_1a['config'] = df_1a['algo'] + ' + ' + df_1a['embed']
df_1a = df_1a.rename(columns={'acc': 'Phase 1-A'})

df_1b = pd.read_csv('phase_1_b/deep_learning_csl_reproduction.csv')
df_1b['config'] = df_1b['algo'] + ' + ' + df_1b['embed']
df_1b = df_1b.rename(columns={'acc': 'Phase 1-B'})

df_2a = pd.read_csv('phase_2_a/ml_baseline_reproduction.csv')
df_2a['config'] = df_2a['algo'] + ' + ' + df_2a['embed']
df_2a = df_2a.rename(columns={'acc': 'Phase 2-A'})

df_2b = pd.read_csv('phase_2_b/csl_ml_baseline_reproduction.csv')
df_2b['config'] = df_2b['algo'] + ' + ' + df_2b['embed']
df_2b = df_2b.rename(columns={'acc': 'Phase 2-B'})

def format_acc(x):
    if pd.isna(x):
        return "-"
    return f"{x*100:.2f}%"

with open('final_stage_comparison.md', 'w', encoding='utf-8') as out:
    out.write("# Comprehensive Cross-Phase Comparison\n\n")
    out.write("This table compares every configuration across the Base Paper, Phase 1-A, Phase 1-B, Phase 2-A, and Phase 2-B.\n\n")
    
    for ds in ['FNFC', 'PROMISE']:
        out.write(f"## Dataset: {ds}\n\n")
        
        df1 = df_1a[df_1a['dataset'] == ds][['config', 'Phase 1-A']] if not df_1a.empty else pd.DataFrame(columns=['config', 'Phase 1-A'])
        df2 = df_1b[df_1b['dataset'] == ds][['config', 'Phase 1-B']] if not df_1b.empty else pd.DataFrame(columns=['config', 'Phase 1-B'])
        df3 = df_2a[df_2a['dataset'] == ds][['config', 'Phase 2-A']] if not df_2a.empty else pd.DataFrame(columns=['config', 'Phase 2-A'])
        df4 = df_2b[df_2b['dataset'] == ds][['config', 'Phase 2-B']] if not df_2b.empty else pd.DataFrame(columns=['config', 'Phase 2-B'])
        
        if not df_base.empty:
            df0 = df_base[df_base['dataset'] == ds][['config', 'Base Paper']]
        else:
            df0 = pd.DataFrame(columns=['config', 'Base Paper'])
            
        merged = pd.merge(df0, df1, on='config', how='outer')
        merged = pd.merge(merged, df2, on='config', how='outer')
        merged = pd.merge(merged, df3, on='config', how='outer')
        merged = pd.merge(merged, df4, on='config', how='outer')
        
        # Sort by Base Paper then Phase 2-A descending
        merged = merged.sort_values(by=['Base Paper', 'Phase 2-B', 'Phase 2-A'], ascending=False)
        
        out.write("| Algorithm + Embedding | Base Paper | Phase 1-A (DL Base) | Phase 1-B (DL CSL) | Phase 2-A (ML Base) | Phase 2-B (Pure CSL) |\n")
        out.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for _, row in merged.iterrows():
            config = row['config']
            v0 = format_acc(row.get('Base Paper', np.nan))
            v1 = format_acc(row.get('Phase 1-A', np.nan))
            v2 = format_acc(row.get('Phase 1-B', np.nan))
            v3 = format_acc(row.get('Phase 2-A', np.nan))
            v4 = format_acc(row.get('Phase 2-B', np.nan))
            out.write(f"| **{config}** | {v0} | {v1} | {v2} | {v3} | {v4} |\n")
        
        out.write("\n---\n\n")
    
    out.write("*Note: Blank values (-) indicate that the algorithm was not run in that specific phase.* \n")
