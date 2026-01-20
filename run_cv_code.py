import os
import subprocess
import pandas as pd

CODE_DIR = "outputs/code"
CSV_DIR = "outputs/csv"

os.makedirs(CSV_DIR, exist_ok=True)

def run_script(script_path: str):
    """
    Executes a generated CV script.
    """
    try:
        subprocess.run(
            ["python", script_path], 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    
def validate_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return False, "CSV file not found"
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"failed to read csv: {e}"
    
    required_cols = {"sample_id", "cv_value"}

    if not required_cols.issubset(df.columns):
        return False, "missing required columns"
    
    if len(df) == 0:
        return False, "empty csv"
    
    if df["cv_value"].isna().any():
        return False, "NaN values for cv_value"
    
    if df["cv_value"].var() < 1e-8:
        return False, "degen CV (near-zero variance)"
    

    return True, None

def main():
    scripts = sorted(
        f for f in os.listdir(CODE_DIR) if f.startswith("compute_") and f.endswith(".py")
    )

    if not scripts:
        raise SystemExit("No scripts")
    

    for script in scripts:
        script_path = os.path.join(CODE_DIR, script)
        cv_name = script.replace("compute_", "").replace(".py", "")
        csv_path = os.path.join(CSV_DIR, f"{cv_name}.csv")

        print(f"\nRunning CV script: {script}")

        success, error = run_script(script_path)

        if not success:
            print(f"Execution failed for {cv_name}: {error}")
            continue

        valid, reason = validate_csv(csv_path)
        if not valid:
            print(f"Validation failed for {cv_name}: {reason}")
            continue

        print(f"CV {cv_name} executed successfully and passed validation")

if __name__ == "__main__":
    main()