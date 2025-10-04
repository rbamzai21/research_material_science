import os
from dotenv import load_dotenv
from llm_utils import query_llm, extract_blocks
from lammps_runner import run_lammps, read_log_tail
from prompts import base_prompt, fix_prompt_with_error

print("Loading environment...")
load_dotenv()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("input3.txt", "r", encoding="utf-8") as f:
    mol_description = f.read()

total_attempts = 15
initial_prompt = base_prompt(mol_description)
prompt = initial_prompt

for i in range(1, total_attempts + 1):
    print(f"\nAttempt-{i}")
    try:
        output = query_llm(prompt, model="gpt-4.1-mini")
    except Exception as e:
        print(f"LLM query failed: {e}")
        break

    with open(f"{OUTPUT_DIR}/llm_output_{i}.txt", "w", encoding="utf-8") as f:
        f.write(output)

    colvars, in_lammps = extract_blocks(output)
    if not colvars or not in_lammps:
        print("Could not extract files from model output.")
        break

    with open(f"{OUTPUT_DIR}/colvars.conf", "w", encoding="utf-8") as f:
        f.write(colvars)
    with open(f"{OUTPUT_DIR}/in.lammps", "w", encoding="utf-8") as f:
        f.write(in_lammps)

    retcode, logfile = run_lammps()
    if retcode == 0:
        print("LAMMPS simulation ran successfully.")
        break

    # debugging part
    print("LAMMPS run failed. Asking LLM to regenerate base prompt...")
    log_tail = read_log_tail(logfile)
    fix_request = fix_prompt_with_error(log_tail, initial_prompt)

    try:
        new_prompt = query_llm(fix_request, model="gpt-4.1-mini")
    except Exception as e:
        print(f"Prompt regeneration failed: {e}")
        continue

    with open(f"{OUTPUT_DIR}/regenerated_prompt_{i}.txt", "w", encoding="utf-8") as f:
        f.write(new_prompt)
    
    print("=== Regenerated Prompt Start ===")
    print(new_prompt)
    print("=== Regenerated Prompt End ===")
   
    if "```tcl" in new_prompt and "```lammps" in new_prompt:
        prompt = new_prompt
        print("Updated prompt with regenerated prompt")
    else:
        print("Not a valid prompt, reverting to base")
        prompt = base_prompt(mol_description)
else:
    print("Could not debug after all attempts")
