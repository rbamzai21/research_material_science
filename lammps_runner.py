import os
import subprocess

OUTPUT_DIR = "outputs"

def run_lammps():
    exe_manya = r"C:\Users\maany\AppData\Local\LAMMPS 64-bit 12Jun2025\bin\lmp.exe"

    log_path = os.path.join(OUTPUT_DIR, "lammps_output.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run([exe_manya, "-in", os.path.join(OUTPUT_DIR, "in.lammps")],
                                stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return result.returncode, log_path

def read_log_tail(log_file, n=40):
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "Log file not found."
    

