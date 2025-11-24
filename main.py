import os
from dotenv import load_dotenv
from llm_utils import query_llm, extract_blocks
from lammps_runner import run_lammps, read_log_tail
from prompts import base_prompt, fix_prompt_with_error

print("Loading environment...")
load_dotenv()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open("input3.txt", "r", encoding="utf-8") as f:
        mol_description = f.read()
except FileNotFoundError:
    raise SystemExit("input3.txt not found")

total_attempts = 15
initial_prompt = base_prompt(mol_description)
prompt = initial_prompt

for i in range(1, total_attempts + 1):
    print(f"\nAttempt-{i}")
    try:
        output = query_llm(prompt, model="gpt-4.1-mini")
    except Exception as e:
        print(f"LLM query failed on attempt {i}: {e}")
        continue

    llm_out_path = os.path.join(OUTPUT_DIR, f"llm_output_{i}.txt")
    try:
        with open(f"{OUTPUT_DIR}/llm_output_{i}.txt", "w", encoding="utf-8") as f:
            f.write(output)
    except Exception as e:
        print("Failed to write LLM output file")

    try:
        colvars, in_lammps = extract_blocks(output)
    except Exception as e:
        print("extract_blocks raised exception")
        continue
    
    if not colvars or not in_lammps or not in_lammps.strip():
        print("Could not extract files from model output.")
        continue

    colvars_path = os.path.join(OUTPUT_DIR, f"colvars_{i}.conf")
    inlammps_path = os.path.join(OUTPUT_DIR, f"in_{i}.lammps")

    try:
        with open(colvars_path, "w", encoding="utf-8") as f:
            f.write(colvars)
        with open(inlammps_path, "w", encoding="utf-8") as f:
            f.write(in_lammps)
        with open(os.path.join(OUTPUT_DIR, "colvars.conf"), "w", encoding="utf-8") as f:
            f.write(colvars)
        with open(os.path.join(OUTPUT_DIR, "in.lammps"), "w", encoding="utf-8") as f:
            f.write(in_lammps)
    except Exception as e:
        print(f"Failed to write extracted files: {e}")
        continue

    try:
        retcode, logfile = run_lammps()
    except Exception as e:
        print(f"run_lammps raised exception: {e}")
        logfile = None
        retcode = -1

    if retcode == 0:
        print("LAMMPS simulation ran successfully.")
        break

    # debugging part
    print("LAMMPS run failed. Asking LLM to regenerate base prompt...")
    log_tail = None
    if logfile:
        try:
            log_tail = read_log_tail(logfile)
        except Exception as e:
            print(f"read_log_tail failed: {e}")
            log_tail = None
    else:
        print("No logfile available to provide to LLM")
    
    fix_request = fix_prompt_with_error(log_tail, initial_prompt)

    try:
        new_prompt = query_llm(fix_request, model="gpt-4.1-mini")
    except Exception as e:
        print(f"Prompt regeneration failed: {e}")
        prompt = base_prompt(mol_description)
        continue

    regen_path = os.path.join(OUTPUT_DIR, f"regenerated_prompt_{i}.txt")
    try:
        with open(regen_path, "w", encoding="utf-8") as f:
            f.write(new_prompt)
    except Exception:
        pass
    
    print("=== Regenerated Prompt Start ===")
    print(new_prompt)
    print("=== Regenerated Prompt End ===")
   
    try:
        regen_colvars, regen_in_lammps = extract_blocks(new_prompt)
    except Exception as e:
        print(f"extract_blocks on regenerated prompt raised: {e}")
        prompt = base_prompt(mol_description)
        continue

    if regen_in_lammps and regen_in_lammps.strip():
        prompt = new_prompt

        try:
            with open(os.path.join(OUTPUT_DIR, f"colvars_regen_{i}.conf"), "w", encoding="utf-8") as f:
                f.write(regen_colvars or "")
            with open(os.path.join(OUTPUT_DIR, f"in_regen_{i}.lammps"), "w", encoding="utf-8") as f:
                f.write(regen_in_lammps or "")
            with open(os.path.join(OUTPUT_DIR, "colvars.conf"), "w", encoding="utf-8") as f:
                f.write(regen_colvars or "")
            with open(os.path.join(OUTPUT_DIR, "in.lammps"), "w", encoding="utf-8") as f:
                f.write(regen_in_lammps or "")
        except Exception as e:
            print(f"Failed to write regenerated files: {e}")
        
        print("Updated prompt with regenerated prompt")
    else:
        print("Not a valid prompt, reverting to base")
        prompt = base_prompt(mol_description)
else:
    print("Could not debug after all attempts")
