import os
import json
from llm_utils import query_llm

CV_DIR = "outputs/cv_specs"
CODE_DIR = "outputs/code"

os.makedirs(CODE_DIR, exist_ok=True)

def code_prompt(cv_spec: dict) -> str:
    """
    Prompt that translates a conceptual CV into executable code.
    """
    return f"""
You are a scientific programmer working with molecular simulation data.

Given the following conceptual collective variable (CV):

{json.dumps(cv_spec, indent=2)}

Write a Python script that computes a reasonable numerical realization
of this CV from molecular simulation data.

Constraints:
- Assume standard molecular simulation trajectory data is available
- Input path: data/input_data
- Output CSV path: outputs/csv/{cv_spec['cv_name']}.csv
- Output CSV columns: sample_id, cv_value
- Use standard scientific Python libraries only (numpy, pandas)
- Make reasonable approximations where necessary
- Do NOT include explanations or comments
- Output ONLY valid executable Python code
"""

def main():
    cv_files = sorted(
        f for f in os.listdir(CV_DIR) if f.endswith(".json")
    )

    if not cv_files:
        raise SystemExit("No CV specs found in outputs/cv_specs")

    for cv_file in cv_files:
        cv_path = os.path.join(CV_DIR, cv_file)

        with open(cv_path, "r", encoding="utf-8") as f:
            cv_spec = json.load(f)

        cv_name = cv_spec.get("cv_name", os.path.splitext(cv_file)[0])
        print(f"Generating code for CV: {cv_name}")

        try:
            code = query_llm(
                prompt=code_prompt(cv_spec),
                system_prompt="You are a scientific programmer.",
                temperature=0.3,
            )
        except Exception as e:
            print(f"LLM failed for {cv_name}: {e}")
            continue

        if "import" not in code or "csv" not in code:
            print(f"Suspicious code output for {cv_name}, skipping")
            continue

        code_path = os.path.join(CODE_DIR, f"compute_{cv_name}.py")
        try:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Saved code to {code_path}")
        except Exception as e:
            print(f"Failed to write code for {cv_name}: {e}")


if __name__ == "__main__":
    main()