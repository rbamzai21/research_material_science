import json


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
  "information_required": [
    "types of data needed (e.g., positions, distances, angles, energies)"
  ],
  "why_it_matters": "why this CV is relevant for this system",
  "definition": "concrete formula"
}}
"""


def code_prompt(cv_spec: dict) -> str:
    """
    Prompt that translates a conceptual CV into executable code.
    """
    return f"""
You are a scientific programmer working with molecular simulation data.

Given the following conceptual collective variable (CV):
{json.dumps(cv_spec, indent=2)}

Write a standalone Python script that computes a numerical realization of this CV.

Requirements:
- DO NOT read any external input files.
- Generate synthetic data internally if needed.
- Write output to: outputs/csv directory
- Ensure the directory outputs/csv exists (use os.makedirs).
- Construct the CSV filename **from the CV name only**, replacing any spaces or special characters with underscores, so it is unique per CV
- Output CSV columns must be exactly: sample_id, rA, rB, rX, t, tau.
- Use only numpy, pandas, os.
- Generate fully executable code.
- Output ONLY valid Python code.
"""
