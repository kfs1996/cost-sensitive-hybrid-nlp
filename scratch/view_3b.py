import pandas as pd

df = pd.read_csv('phase_3_b/hybrid_csl_ml_reproduction.csv')
df_avg = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()

# Format as percentages
df_avg['acc'] = (df_avg['acc'] * 100).round(2).astype(str) + '%'

fnfc = df_avg[df_avg['dataset']=='FNFC'].pivot(index='algo', columns='embed', values='acc')
prom = df_avg[df_avg['dataset']=='PROMISE'].pivot(index='algo', columns='embed', values='acc')

print("FNFC (14-class):")
print(fnfc.to_markdown())
print("\nPROMISE (12-class):")
print(prom.to_markdown())
