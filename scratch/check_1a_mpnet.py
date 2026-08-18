import pandas as pd
df = pd.read_csv('phase_1_a/deep_learning_baseline_reproduction.csv')
mpnet = df[df['embed'] == 'MPNet']
print(mpnet.groupby(['dataset', 'algo'])['acc'].mean())
