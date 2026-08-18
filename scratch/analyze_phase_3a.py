import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
phase3a_file = _ROOT / "phase_3_a" / "hybrid_dl_baseline_reproduction.csv"

def generate_sorted_tables():
    if not phase3a_file.exists():
        print(f"Error: {phase3a_file} does not exist yet. Phase 3-A must finish running first.")
        return

    print("Loading Phase 3-A results...")
    df = pd.read_csv(phase3a_file)

    # Average the accuracy across the 5 folds
    df_avg = df.groupby(['dataset', 'algo', 'embed'])['acc'].mean().reset_index()

    # Split by dataset and sort from highest to lowest
    fnfc_df = df_avg[df_avg['dataset'] == 'FNFC'].sort_values(by='acc', ascending=False)
    prom_df = df_avg[df_avg['dataset'] == 'PROMISE'].sort_values(by='acc', ascending=False)

    # Output formatted markdown files
    out_dir = _ROOT / "phase_3_a"
    
    fnfc_df.to_csv(out_dir / "fnfc_sorted_results.csv", index=False)
    prom_df.to_csv(out_dir / "promise_sorted_results.csv", index=False)

    # Generate Markdown Table
    md_content = "# Phase 3-A: Hybrid Embeddings Results (Sorted Highest to Lowest)\n\n"
    
    md_content += "## Dataset: FNFC\n"
    md_content += "| Rank | Embedding Combination | Deep Learning Algorithm | Cross-Validated Accuracy |\n"
    md_content += "|---|---|---|---|\n"
    for rank, row in enumerate(fnfc_df.itertuples(), 1):
        md_content += f"| {rank} | {row.embed} | {row.algo} | {row.acc:.4f} |\n"
        
    md_content += "\n## Dataset: PROMISE\n"
    md_content += "| Rank | Embedding Combination | Deep Learning Algorithm | Cross-Validated Accuracy |\n"
    md_content += "|---|---|---|---|\n"
    for rank, row in enumerate(prom_df.itertuples(), 1):
        md_content += f"| {rank} | {row.embed} | {row.algo} | {row.acc:.4f} |\n"

    with open(out_dir / "phase_3a_sorted_results.md", "w") as f:
        f.write(md_content)

    print("Sorted results successfully generated!")
    print(f"FNFC Absolute Max: {fnfc_df.iloc[0]['acc']:.4f} -> {fnfc_df.iloc[0]['embed']} + {fnfc_df.iloc[0]['algo']}")
    print(f"PROMISE Absolute Max: {prom_df.iloc[0]['acc']:.4f} -> {prom_df.iloc[0]['embed']} + {prom_df.iloc[0]['algo']}")

if __name__ == "__main__":
    generate_sorted_tables()
