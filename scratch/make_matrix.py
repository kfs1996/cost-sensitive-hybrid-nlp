import pandas as pd

df = pd.read_csv('phase_2_b/csl_ml_baseline_reproduction.csv')
fnfc = df[df['dataset'] == 'FNFC']

# Pivot table: rows = algo, cols = embed
pivot = fnfc.pivot(index='algo', columns='embed', values='acc')

# Reorder columns logically
cols = ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'SBERT', 'MPNet']
pivot = pivot[cols]

# Format as percentages
pivot = pivot.applymap(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")

print(pivot.to_markdown())
