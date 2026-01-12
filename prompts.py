def base_prompt(desc):
    return f"""
You are a molecular simulation expert.

Given the following system description:
\"\"\"{desc}\"\"\"

Your tasks is to suggest 1 meaningful collective variables (CVs) based on the molecular structure.

You must output this result in the following JSON format and nothing else: 

{{
  "cv_name": "short descriptive name",
  "physical_quantity": "what physical phenomenon this CV measures",
  "high_level_definition": "conceptual or mathematical description, no implementation details",
  "information_required": [
    "types of data needed (e.g., positions, distances, angles, energies)"
  ],
  "why_it_matters": "why this CV is relevant for this system"
}}
"""

def code_prompt(cv_json):
    return f"""
You are a scientific programmer. 

Given the following conceptual collective variable definition:
\"\"\"{cv_json}\"\"\"

Write a Python script that computes a reasonable numerical realization of this CV from molecular simulation data. 

Constraints:
- Input path: data/input_data
- Output CSV path: outputs/csv/cv_values.csv
- Output columns: sample_id, cv_value
- Use standard specific Python libraries only
- Make reasonable approximations if needed
- Output ONLY executable python code
"""


