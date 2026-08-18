import os
import subprocess

scripts = [
    "phase_2_a/run_28_ml_baselines.py",
    "phase_2_b/run_28_csl_ml_baselines.py",
    "phase_3_b/run_phase_3b.py",
    "phase_3_c/run_phase_3c.py"
]

print("Starting Plot Generation for all Classical ML Phases (2-A, 2-B, 3-B, 3-C)")

for script in scripts:
    print(f"\n======================\nRunning {script}...")
    subprocess.run(["C:\\ProgramData\\anaconda3\\python.exe", script])
    print(f"Finished {script}")

print("\n\nAll ML Plots Generated Successfully!")
