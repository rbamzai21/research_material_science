import json
import os

from dotenv import load_dotenv
from llm_utils import query_llm

from prompts import base_prompt

print("Loading environment...")
load_dotenv()

OUTPUT_DIR = "outputs"
CV_DIR = os.path.join(OUTPUT_DIR, "cv_specs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open("input3.txt", encoding="utf-8") as f:
        mol_description = f.read()
except FileNotFoundError:
    raise SystemExit("input3.txt not found")

total_attempts = 3
prompt = base_prompt(mol_description)

for i in range(1, total_attempts + 1):
    print(f"\nAttempt-{i}")

    try:
        output = query_llm(prompt, model="gpt-4.1-mini")
    except Exception as e:
        print(f"LLM query failed on attempt {i}: {e}")
        continue

    # parse json
    try:
        cv_spec = json.loads(output)
    except json.JSONDecodeError:
        print(f"Attempt {i}: Invalid JSON output")
        continue

    # validate
    required_keys = {
        "cv_name",
        "physical_quantity",
        "definition",
        "information_required",
        "why_it_matters",
    }

    if not required_keys.issubset(cv_spec):
        print(f"Attempt {i}: missing required cv fields, skipping")
        continue

    # save CV artifact
    cv_path = os.path.join(OUTPUT_DIR, f"cv_{i}.json")
    try:
        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(cv_spec, f, indent=2)
        print(f"Save CV spec to {cv_path}")
    except Exception as e:
        print(f"Failed to write CV file: {e}")
