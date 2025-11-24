import os
import subprocess
from lammps import lammps

OUTPUT_DIR = "outputs"

def run_lammps():
    input_path = os.path.join(OUTPUT_DIR, "in.lammps")
    log_path = os.path.join(OUTPUT_DIR, "lammps_output.log")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    with open(log_path, "w", encoding="utf-8") as log_file:
        lmp = lammps()

        if hasattr(lmp, "file"):
            lmp.file(os.path.abspath(input_path))
        else:
            with open(input_path, "r", encoding="utf-8"):
                for r in f:
                    line = r.strip()
                    if not line or line.startswith("#"):
                        continue
                    lmp.command(line)
        print("LAMMPS run finished.", file=log_file)

    return 0, log_path

def read_log_tail(log_file, n=40):
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "Log file not found."
    
if __name__ == "__main__":
    code, log = run_lammps()
    print(f"Finished with return code {code}. Log: {log}")
    print(read_log_tail(log))
    

