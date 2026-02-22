import os
import subprocess
import sys
import tempfile

import pandas as pd

CODE_DIR = "outputs/code"
CSV_DIR = "outputs/csv"

os.makedirs(CSV_DIR, exist_ok=True)


def run_script(script_path: str):
    # Read original script
    with open(script_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Remove first line if it's literally 'python'
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]

    # Prepend UTF-8 encoding declaration
    lines = ["# -*- coding: utf-8 -*-\n"] + lines

    # Write cleaned script to temp file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.writelines(lines)
        temp_script_path = tmp.name

    try:
        subprocess.run(
            [sys.executable, temp_script_path],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)


def validate_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return False, "CSV file not found"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"failed to read csv: {e}"

    required_cols = {"sample_id", "rA", "rB", "rX", "t", "tau"}

    if not required_cols.issubset(df.columns):
        return False, "missing required columns"

    if len(df) == 0:
        return False, "empty csv"

    return True, None


def main():
    scripts = sorted(
        f for f in os.listdir(CODE_DIR) if f.startswith("compute_") and f.endswith(".py")
    )

    if not scripts:
        raise SystemExit("No scripts found in outputs/code")

    for script in scripts:
        script_path = os.path.join(CODE_DIR, script)
        cv_name = script.replace("compute_", "").split("_", 1)[1].replace(".py", "")

        print(f"\nRunning CV script: {script}")

        success, error = run_script(script_path)

        if not success:
            print(f"Execution failed for {cv_name}: {error}")
            continue

        print(f"CV {cv_name} executed successfully")


if __name__ == "__main__":
    main()
