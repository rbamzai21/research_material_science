def base_prompt(desc):
    return f"""
You are a molecular simulation expert.

Given the following system description:

\"\"\"{desc}\"\"\"

Your tasks are:
1. Suggest 3 meaningful collective variables (CVs) based on the molecular structure.
2. Write a valid `colvars.conf` file for LAMMPS using the Colvars module. Always define `atomGroup` sections internally within this file — do not assume LAMMPS group definitions.
3. Write a valid LAMMPS input file `in.lammps` that:
   - Uses `units real`
   - Reads a given `.data` file containing the molecular system
   - Applies the Colvars fix using the following command:

     fix mycv all colvars outputs/colvars.conf

Important Rules:
- In `colvars.conf`, use `atomGroup` blocks to define all atom groups with explicit atom IDs.
- Never refer to LAMMPS groups (e.g., `group1`, `all_atoms`) inside `colvars.conf`.
- Do not include any explanation or commentary outside the code blocks.

Formatting Instructions:
- Enclose the `colvars.conf` content in triple backticks marked with `tcl`: ```tcl
- Enclose the `in.lammps` content in triple backticks marked with `lammps`: ```lammps
- Output only the two files. Do not include any other text.

Output format:
```tcl
# colvars.conf
...
```
```lammps
# in.lammps
...
```
"""


def fix_prompt_with_error(log_tail, base):
    return f"""
A LAMMPS simulation failed with the following error log:
```

Regenerate a corrected version of the prompt **strictly in the same format** as the base.

Regenerate a corrected version of the prompt **strictly in the same format** as the base. 

Mandatory rules:
- Output must begin with the line: "You are a molecular simulation expert."
- Include both ```tcl and ```lammps code blocks.
- Do not include any explanation or commentary outside the code blocks.

Here is the original prompt:

\"\"\"{base}\"\"\"
"""
