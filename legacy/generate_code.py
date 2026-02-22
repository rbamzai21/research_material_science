import json
import os

from llm_utils import query_llm

from prompts import code_prompt

CV_DIR = "outputs/"
CODE_DIR = "outputs/code"

os.makedirs(CODE_DIR, exist_ok=True)


def main():
    cv_files = sorted(f for f in os.listdir(CV_DIR) if f.endswith(".json"))

    if not cv_files:
        raise SystemExit("No CV specs found in outputs/")

    for i, cv_file in enumerate(cv_files):
        cv_path = os.path.join(CV_DIR, cv_file)

        with open(cv_path, encoding="utf-8") as f:
            cv_spec = json.load(f)

        cv_name = os.path.splitext(cv_file)[0]
        print(f"Generating code for CV: {cv_name}")

        try:
            code = query_llm(
                prompt=code_prompt(cv_spec),
                system_prompt="You are a scientific programmer.",
            )

            code = strip_code_fences(code)
        except Exception as e:
            print(f"LLM failed for {cv_name}: {e}")
            continue

        if "import" not in code or "csv" not in code:
            print(f"Suspicious code output for {cv_name}, skipping")
            continue

        code_path = os.path.join(CODE_DIR, f"compute_{i}_{cv_name}.py")
        try:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Saved code to {code_path}")
        except Exception as e:
            print(f"Failed to write code for {cv_name}: {e}")


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


if __name__ == "__main__":
    main()
