import pandas as pd
import os
from pathlib import Path

_ROOT = Path('.')
out_path = _ROOT / "Consolidated_Phase_Results.xlsx"

phases = {
    "Phase 1-A": _ROOT / "outputs" / "results" / "deep_learning_baseline_reproduction.csv",
    "Phase 1-B": _ROOT / "phase_1_b" / "deep_learning_csl_reproduction.csv",
    "Phase 2-A": _ROOT / "phase_2_a" / "ml_baseline_reproduction.csv",
    "Phase 2-B": _ROOT / "phase_2_b" / "csl_ml_baseline_reproduction.csv",
    "Phase 3-B": _ROOT / "phase_3_b" / "hybrid_csl_ml_reproduction.csv",
    "Phase 3-C": _ROOT / "phase_3_c" / "tri_hybrid_csl_ml_reproduction.csv"
}

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for phase_name, csv_file in phases.items():
        if not csv_file.exists():
            print(f"Skipping {phase_name}, file not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        
        # Aggregate across folds to remove fold-wise data
        if 'fold' in df.columns:
            df = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()
            
        current_row = 0
        worksheet = writer.book.create_sheet(phase_name)
        writer.sheets[phase_name] = worksheet
        
        for ds in ['FNFC', 'PROMISE']:
            ds_df = df[df['dataset'] == ds]
            if ds_df.empty:
                continue
                
            pivot = ds_df.pivot(index='algo', columns='embed', values='acc')
            
            # Format as percentage
            pivot = pivot.map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
            
            # Write Title
            worksheet.cell(row=current_row + 1, column=1, value=f"{ds} Dataset")
            current_row += 1
            
            # Write DataFrame
            pivot.to_excel(writer, sheet_name=phase_name, startrow=current_row)
            current_row += len(pivot) + 4 # Add spacing for next table

# Remove default empty sheet
if 'Sheet' in writer.book.sheetnames:
    del writer.book['Sheet']

print(f"Successfully generated {out_path}!")
