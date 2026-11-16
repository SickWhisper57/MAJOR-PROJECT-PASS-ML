"""Run the whole pipeline: dataset -> training -> evaluation -> figures."""
import subprocess, sys
for step in ["s01_generate_dataset.py", "s02_train_model.py", "s03_evaluate.py", "s04_make_figures.py"]:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, step], check=True)
