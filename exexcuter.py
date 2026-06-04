import subprocess
import sys

scripts = [
    "generate_data.py",
    "train_model.py",
    "run_quantum.py",
    "benchmark.py",
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run([sys.executable, script], check=True)
