import pandas as pd

df = pd.read_csv('phase_2_b/csl_ml_baseline_reproduction.csv')
df_avg = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()

fnfc = df_avg[df_avg['dataset']=='FNFC'].pivot(index='algo', columns='embed', values='acc')
prom = df_avg[df_avg['dataset']=='PROMISE'].pivot(index='algo', columns='embed', values='acc')

print("FNFC:")
print(fnfc.to_markdown())
print("\nPROMISE:")
print(prom.to_markdown())
