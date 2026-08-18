import pandas as pd

try:
    df = pd.read_csv('phase_3_c/tri_hybrid_csl_ml_reproduction.csv')
    counts = df.groupby(['dataset', 'algo', 'embed']).size()
    
    # Filter only those that have finished all 5 folds
    completed = counts[counts == 5].reset_index()
    
    if len(completed) == 0:
        print("No algorithm has completely finished all 5 folds yet.")
        print(f"Current total evaluations in CSV: {len(df)}")
    else:
        df_complete = df.merge(completed[['dataset', 'algo', 'embed']], on=['dataset', 'algo', 'embed'])
        df_avg = df_complete.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()
        
        # Format as percentages
        df_avg['acc'] = (df_avg['acc'] * 100).round(2).astype(str) + '%'
        
        # We might not have all algorithms, so just print what we have nicely
        fnfc = df_avg[df_avg['dataset']=='FNFC']
        prom = df_avg[df_avg['dataset']=='PROMISE']
        
        if not fnfc.empty:
            print("FNFC Completed (5-Folds Final Average):")
            print(fnfc.pivot(index='algo', columns='embed', values='acc').to_markdown())
            
        if not prom.empty:
            print("\nPROMISE Completed (5-Folds Final Average):")
            print(prom.pivot(index='algo', columns='embed', values='acc').to_markdown())
            
except Exception as e:
    print(f"Error reading CSV: {e}")
